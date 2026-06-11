import argparse
import SimpleITK as sitk
import numpy as np
import sys
import os
import math

def initial_tissue_mask(img, upper_thresh, gen_file):
    """
    Generates a binary mask of the largest solid tissue component in the image.

    Args:
        img (SimpleITK.Image): The original 3D image scan.
        upper_thresh (float): The threshold value to separate solid tissue from air.
        gen_file (bool): Flag to generate a debug NIfTI file of the mask.

    Returns:
        numpy.ndarray: A 3D numpy array representing the largest solid tissue mask.
    """
    print("[INFO] Generating initial solid tissue mask...")
    solid_mask = img > upper_thresh
    solid_components = sitk.ConnectedComponent(solid_mask)
    largest_solid = sitk.RelabelComponent(solid_components) == 1

    if gen_file:
        print("  [DEBUG] Generating step1_initial_tissue_body.nii.gz...")
        sitk.WriteImage(sitk.Cast(largest_solid, sitk.sitkUInt8),
                        "step1_initial_tissue_body.nii.gz")

    return sitk.GetArrayFromImage(largest_solid)

def extract_middle(solid_array):
    """
    Extracts the middle Z-slice from the solid mask to use as a 2D reference blueprint.

    Args:
        solid_array (numpy.ndarray): The 3D solid tissue mask array.

    Returns:
        tuple: A tuple containing (min_x, max_x, min_y, max_y, reference_slice),
               defining the bounding box and the 2D array of the middle slice.
    """
    print("[INFO] Extracting the middle Z-slice as the clean reference blueprint...")
    mid_z = solid_array.shape[0] // 2
    reference_slice = solid_array[mid_z, :, :]

    valid_y, valid_x = np.where(reference_slice)
    if len(valid_y) == 0 or len(valid_x) == 0:
        print("[ERROR] No solid phantom tissue detected on the middle slice.")
        sys.exit(1)

    min_x, max_x = np.min(valid_x), np.max(valid_x)
    min_y, max_y = np.min(valid_y), np.max(valid_y)

    return min_x, max_x, min_y, max_y, reference_slice

def calc_crop(min_x, max_x, min_y, max_y, reference_slice, tissue_to_air):
    """
    Calculates tight crop boundaries based on a tissue-to-air density ratio.

    Args:
        min_x (int): Minimum X coordinate from the reference bounding box.
        max_x (int): Maximum X coordinate from the reference bounding box.
        min_y (int): Minimum Y coordinate from the reference bounding box.
        max_y (int): Maximum Y coordinate from the reference bounding box.
        reference_slice (numpy.ndarray): The 2D middle slice used for calculations.
        tissue_to_air (float): The required ratio of tissue to air to trigger a crop boundary.

    Returns:
        tuple: A tuple containing (crop_top, crop_bottom, crop_left, crop_right).
    """
    print(f"[INFO] Calculating boundaries using the 4-sided {tissue_to_air} tissue rule...")
    crop_bottom = int(max_y)
    for y in range(max_y, min_y - 1, -1):
        row_segment = reference_slice[y, min_x:max_x + 1]
        tissue_ratio = np.sum(row_segment) / len(row_segment)
        if tissue_ratio > tissue_to_air + 0.03:
            crop_bottom = int(y)
            break

    crop_top = int(min_y)
    for y in range(min_y, crop_bottom + 1):
        row_segment = reference_slice[y, min_x:max_x + 1]
        tissue_ratio = np.sum(row_segment) / len(row_segment)
        if tissue_ratio > tissue_to_air - 0.035:
            crop_top = int(y)
            break

    crop_left = int(min_x)
    for x in range(min_x, max_x + 1):
        column_segment = reference_slice[crop_top:crop_bottom + 1, x]
        tissue_ratio = np.sum(column_segment) / len(column_segment)
        if tissue_ratio > tissue_to_air:
            crop_left = int(x)
            break

    crop_right = int(max_x)
    for x in range(max_x, min_x - 1, -1):
        column_segment = reference_slice[crop_top:crop_bottom + 1, x]
        tissue_ratio = np.sum(column_segment) / len(column_segment)
        if tissue_ratio > tissue_to_air:
            crop_right = int(x)
            break

    return crop_top, crop_bottom, crop_left, crop_right

def perform_crop(img, img_array, crop_top, crop_bottom, crop_left, crop_right, gen_file):
    """
    Crops the original 3D image using the computed 2D boundaries.

    Args:
        img (SimpleITK.Image): The original 3D scan with physical metadata.
        img_array (numpy.ndarray): The raw data array of the original scan.
        crop_top (int): Top boundary (Y-axis).
        crop_bottom (int): Bottom boundary (Y-axis).
        crop_left (int): Left boundary (X-axis).
        crop_right (int): Right boundary (X-axis).
        gen_file (bool): Flag to generate a debug NIfTI file.

    Returns:
        SimpleITK.Image: The cropped 3D image with preserved spatial metadata.
    """
    print(f"[INFO] Executing 2D Crop: X({crop_left} to {crop_right}), "
          f"Y({crop_top} to {crop_bottom})")
    cropped_array = img_array[:, crop_top:crop_bottom + 1, crop_left:crop_right + 1]

    cropped_img = sitk.GetImageFromArray(cropped_array)
    cropped_img.SetSpacing(img.GetSpacing())
    cropped_img.SetDirection(img.GetDirection())
    cropped_img.SetOrigin(img.TransformIndexToPhysicalPoint([crop_left, crop_top, 0]))

    if gen_file:
        print("  [DEBUG] Generating step4_cropped_raw_image.nii.gz...")
        sitk.WriteImage(cropped_img, "step4_cropped_raw_image.nii.gz")

    return cropped_img

def segment_air(cropped_img, upper_thresh, gen_file):
    """
    Isolates trapped internal air pockets within the cropped tissue volume.

    Args:
        cropped_img (SimpleITK.Image): The cropped 3D image.
        upper_thresh (float): Threshold to differentiate air from tissue.
        gen_file (bool): Flag to generate a debug NIfTI file.

    Returns:
        SimpleITK.Image: A 3D binary mask of the internal air.
    """
    print("[INFO] Segmenting trapped air within the cropped boundaries...")
    cropped_tissue_mask = cropped_img > upper_thresh
    padded_tissue = sitk.ConstantPad(cropped_tissue_mask, [1, 1, 1], [1, 1, 1], 1)

    filled_tissue = sitk.BinaryFillhole(padded_tissue)
    sealed_shell = sitk.Crop(filled_tissue, [1, 1, 1], [1, 1, 1])

    air_mask = cropped_img <= upper_thresh
    internal_air = air_mask * sealed_shell

    if gen_file:
        print("  [DEBUG] Generating step5_cropped_all_internal_air.nii.gz...")
        sitk.WriteImage(sitk.Cast(internal_air, sitk.sitkUInt8),
                        "step5_cropped_all_internal_air.nii.gz")

    return internal_air

def calc_top_shave(internal_air, target_circularity, shave_limit):
    """
    Analyzes slices from the top downward, calculating how many slices to shave
    based on shape circularity to remove artifacts.

    Args:
        internal_air (SimpleITK.Image): Binary mask of internal air pockets.
        target_circularity (float): The minimum circularity score required to stop shaving.
        shave_limit (int): The maximum number of slices allowed to be shaved.

    Returns:
        int: The number of slices to shave from the top (Z-axis).
    """
    print(f"[INFO] Analyzing top slices for Circularity (Target >= {target_circularity})...")
    depth = internal_air.GetDepth()
    image_width = internal_air.GetWidth()
    image_height = internal_air.GetHeight()
    shave_count = 0

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)

    for z in range(depth -1, -1, -1):
        slice_2d = internal_air[:, :, z]
        cc_2d = cc_filter.Execute(slice_2d)
        largest_2d_component = sitk.RelabelComponent(cc_2d)
        shape_stats.Execute(largest_2d_component)

        if not shape_stats.HasLabel(1):
            shave_count += 1
            continue

        bbox = shape_stats.GetBoundingBox(1)
        bbox_width = bbox[2]
        bbox_height = bbox[3]

        if bbox_width > 0.80 * image_width or bbox_height > 0.80 * image_height:
            shave_count += 1
            continue

        area = shape_stats.GetPhysicalSize(1)
        perimeter = shape_stats.GetPerimeter(1)

        if perimeter == 0:
            shave_count += 1
            continue

        circularity = (4 * math.pi * area) / (perimeter ** 2)

        if circularity < target_circularity:
            shave_count += 1
        else:
            print(f"  -> [SUCCESS] Found valid circular lumen at depth {shave_count} "
                  f"(Circularity: {circularity:.2f})")
            break

    if shave_count == depth:
        print("  -> [WARNING] No top slice met the circularity target. Reverting to 0 shave.")
        return 0
    elif shave_count > shave_limit:
        print(f"  -> [WARNING] Excessive depth, shave capped at {shave_limit}.")
        return shave_limit

    return shave_count

def calc_bottom_shave(internal_air, target_extent, shave_limit):
    """
    Analyzes slices from the bottom upward, calculating how many slices to shave
    based on elliptical extent to remove artifacts.

    Args:
        internal_air (SimpleITK.Image): Binary mask of internal air pockets.
        target_extent (float): The minimum extent score required to stop shaving.
        shave_limit (int): The maximum number of slices allowed to be shaved.

    Returns:
        int: The number of slices to shave from the bottom (Z-axis).
    """
    print(f"[INFO] Analyzing bottom slices for elliptical Extent (Target >= {target_extent})...")
    depth = internal_air.GetDepth()
    image_width = internal_air.GetWidth()
    image_height = internal_air.GetHeight()
    shave_count = 0

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)

    for z in range(0, depth):
        slice_2d = internal_air[:, :, z]
        cc_2d = cc_filter.Execute(slice_2d)
        largest_2d = sitk.RelabelComponent(cc_2d)
        shape_stats.Execute(largest_2d)

        if not shape_stats.HasLabel(1):
            shave_count += 1
            continue

        bbox = shape_stats.GetBoundingBox(1)
        bbox_width = bbox[2]
        bbox_height = bbox[3]

        if bbox_width > 0.80 * image_width or bbox_height > 0.80 * image_height:
            shave_count += 1
            continue

        if bbox_height > 0:
            aspect_ratio = bbox_width / bbox_height
            if aspect_ratio > 4.0:
                shave_count += 1
                continue

        area_pixels = shape_stats.GetNumberOfPixels(1)
        bbox_area = bbox_width * bbox_height

        if bbox_area == 0:
            shave_count += 1
            continue

        extent = area_pixels / bbox_area

        if extent < target_extent:
            shave_count += 1
        else:
            print(f"  -> [SUCCESS] Hit valid elliptical lumen. "
                  f"Dropping bottom {shave_count} slices (Score: {extent:.2f})")
            break

    if shave_count == depth:
        print("  -> [WARNING] No bottom slice met the extent target. Reverting to 0 shave.")
        return 0
    elif shave_count > shave_limit:
        print(f"  -> [WARNING] Excessive depth, shave capped at {shave_limit}.")
        return shave_limit

    return shave_count

def calc_anterior_shave(internal_air, target_circularity, shave_limit):
    """
    Analyzes anterior coronal slices inward to calculate anterior face shaving based on circularity.

    Args:
        internal_air (SimpleITK.Image): Binary mask of internal air pockets.
        target_circularity (float): Minimum circularity target for coronal slices.
        shave_limit (int): Maximum number of slices allowed to be shaved.

    Returns:
        int: The number of slices to shave from the anterior face (Y-axis).
    """
    print(f"[INFO] Analyzing Anterior (-Y) slices for semi-circular shape "
          f"(Target >= {target_circularity})...")
    image_width = internal_air.GetWidth()
    image_height = internal_air.GetHeight()
    image_depth = internal_air.GetDepth()
    shave_count = 0

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)

    for y in range(0, image_height):
        slice_2d = internal_air[:, y, :]
        cc_2d = cc_filter.Execute(slice_2d)
        largest_2d = sitk.RelabelComponent(cc_2d)
        shape_stats.Execute(largest_2d)

        if not shape_stats.HasLabel(1):
            shave_count += 1
            continue

        bbox = shape_stats.GetBoundingBox(1)
        bbox_w = bbox[2]
        bbox_h = bbox[3]

        if bbox_w > 0.80 * image_width or bbox_h > 0.80 * image_depth:
            shave_count += 1
            continue

        area = shape_stats.GetPhysicalSize(1)
        perimeter = shape_stats.GetPerimeter(1)

        if perimeter == 0:
            shave_count += 1
            continue

        circularity = (4 * math.pi * area) / (perimeter ** 2)

        if circularity < target_circularity:
            shave_count += 1
        else:
            print(f"  -> [SUCCESS] Found valid semi-circular lumen at Y-depth {shave_count} "
                  f"(Circularity: {circularity:.2f})")
            break

    if shave_count == image_height:
        print("  -> [WARNING] No anterior slice met the target. Reverting to 0 shave.")
        return 0
    elif shave_count > shave_limit:
        print(f"  -> [WARNING] Excessive depth. Shave capped at {shave_limit}.")
        return shave_limit

    return shave_count

def shave_faces(internal_air, shave_global, dynamic_shave_top, dynamic_shave_bottom,
                dynamic_shave_anterior, gen_file):
    """
    Applies calculated asymmetric crop bounds to the air mask and isolates the largest component.

    Args:
        internal_air (SimpleITK.Image): Binary mask of internal air pockets.
        shave_global (int): Base uniform voxel shave applied to remaining non-dynamic faces.
        dynamic_shave_top (int): Calculated shave depth for the top face.
        dynamic_shave_bottom (int): Calculated shave depth for the bottom face.
        dynamic_shave_anterior (int): Calculated shave depth for the anterior face.
        gen_file (bool): Flag to generate a debug NIfTI file.

    Returns:
        SimpleITK.Image: A 3D mask containing the final, isolated continuous lumen.
    """
    z_shave_bottom = dynamic_shave_bottom
    z_shave_top = dynamic_shave_top
    x_shave_left = shave_global
    x_shave_right = shave_global
    y_shave_anterior = dynamic_shave_anterior
    y_shave_posterior = shave_global

    if z_shave_bottom + z_shave_top >= internal_air.GetDepth():
        print("  -> [WARNING] Combined Z-shaves exceed image depth! Reverting Z-shave to 0.")
        z_shave_bottom = 0
        z_shave_top = 0

    if y_shave_anterior + y_shave_posterior >= internal_air.GetHeight():
        print("  -> [WARNING] Combined Y-shaves exceed image height! Reverting Y-shave to 0.")
        y_shave_anterior = 0
        y_shave_posterior = 0

    lower_shave_bounds = [x_shave_left, y_shave_anterior, z_shave_bottom]
    upper_shave_bounds = [x_shave_right, y_shave_posterior, z_shave_top]

    print(f"[INFO] Applying Boundary Erosion -> Lower [X,Y,Z]: {lower_shave_bounds}, "
          f"Upper [X,Y,Z]: {upper_shave_bounds}")
    shaved_air = sitk.Crop(internal_air, lower_shave_bounds, upper_shave_bounds)

    disconnected_air = sitk.ConstantPad(shaved_air, lower_shave_bounds, upper_shave_bounds, 0)

    print("[INFO] Filtering disconnected components...")
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)
    air_components = cc_filter.Execute(disconnected_air)
    largest_lumen = sitk.RelabelComponent(air_components) == 1

    if gen_file:
        print("  [DEBUG] Generating step6_cropped_isolated_lumen.nii.gz...")
        sitk.WriteImage(sitk.Cast(largest_lumen, sitk.sitkUInt8),
                        "step6_cropped_isolated_lumen.nii.gz")

    return largest_lumen

def auto_segment_lumen(input_path, output_path, upper_thresh, shave_global, target_top,
                       target_bottom, target_anterior, shave_limit, gen_file, target_label,
                       tissue_to_air):
    """
    Orchestrates the full pipeline to read a scan, segment the main lumen, and write the mask.

    Args:
        input_path (str): Filepath to the raw input scan.
        output_path (str): Filepath to write the final lumen mask.
        upper_thresh (float): Threshold to partition tissue from air.
        shave_global (int): Base uniform shave parameter.
        target_top (float): Circularity target for top-down dynamic shaving.
        target_bottom (float): Extent target for bottom-up dynamic shaving.
        target_anterior (float): Circularity target for anterior dynamic shaving.
        shave_limit (int): Cap for dynamic shaving slice removal.
        gen_file (bool): Toggles debug intermediate file generation.
        target_label (int): Integer label written to the final output mask.
        tissue_to_air (float): Density ratio rule for cropping operations.
    """
    if not os.path.exists(input_path):
        print(f"\n[ERROR] The input scan '{input_path}' was not found.")
        sys.exit(1)

    try:
        print(f"\n[INFO] Loading raw scan '{input_path}'...")
        img = sitk.ReadImage(input_path)
        img_array = sitk.GetArrayFromImage(img)

        print("\n=== STEP 1: INITIAL TISSUE MASK ===")
        solid_array = initial_tissue_mask(img, upper_thresh, gen_file)

        print("\n=== STEP 2: THE 'GOLDEN SLICE' REFERENCE ===")
        min_x, max_x, min_y, max_y, reference_slice = extract_middle(solid_array)

        print("\n=== STEP 3: CALCULATE 4-WAY CROP LINES ===")
        crop_top, crop_bottom, crop_left, crop_right = (
            calc_crop(min_x, max_x, min_y, max_y, reference_slice, tissue_to_air))

        print("\n=== STEP 4: EXECUTE THE 3D CROP ===")
        cropped_img = perform_crop(img, img_array, crop_top, crop_bottom, crop_left, crop_right,
                                   gen_file)

        print("\n=== STEP 5: SEGMENT INTERNAL AIR ===")
        internal_air = segment_air(cropped_img, upper_thresh, gen_file)

        print("\n=== STEP 6: ASYMMETRICALLY SHAVE SIDES ===")
        dynamic_shave_top = calc_top_shave(internal_air, target_top, shave_limit)
        dynamic_shave_bottom = calc_bottom_shave(internal_air, target_bottom, shave_limit)
        dynamic_shave_anterior = calc_anterior_shave(internal_air, target_anterior, shave_limit)
        shaved_lumen = shave_faces(internal_air, shave_global, dynamic_shave_top,
                                    dynamic_shave_bottom, dynamic_shave_anterior, gen_file)

        print("\n=== STEP 7: PASTE BACK TO ORIGINAL DIMENSIONS ===")
        print("[INFO] Realigning mask with original dimensions...")
        final_full_mask = sitk.Image(img.GetSize(), sitk.sitkUInt8)
        final_full_mask.CopyInformation(img)

        lumen_mask_8bit = sitk.Cast(shaved_lumen, sitk.sitkUInt8) * target_label
        final_full_mask = sitk.Paste(
            destinationImage=final_full_mask,
            sourceImage=lumen_mask_8bit,
            sourceSize=lumen_mask_8bit.GetSize(),
            sourceIndex=[0, 0, 0],
            destinationIndex=[crop_left, crop_top, 0]
        )

        print(f"\n[INFO] Saving final result to '{output_path}'...")
        sitk.WriteImage(final_full_mask, output_path)
        print("[INFO] Processing complete!\n")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during processing: {e}\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive BPH Lumen Segmentation.")

    parser.add_argument("-i", "--input", required=True,
                        help="Path to raw scan (REQUIRED)")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to save final mask (REQUIRED)")
    parser.add_argument("-l", "--label", type=int, default=2,
                        help="Int; Label number to work on (default: 2)")
    parser.add_argument("-u", "--upper_thresh", type=float, default=-500.0,
                        help="Float; Upper threshold limit (default: -500)")
    parser.add_argument("-t", "--tissue_to_air", type=float, default=0.92,
                        help="Float; Tissue-to-air ratio limit (default: 0.92)")
    parser.add_argument("-s", "--shave", type=int, default=5,
                        help="Int; Global number of voxels to shave off all 6 faces (default: 5)")
    parser.add_argument("-st", "--shave_top", type=float, default=0.5,
                        help="Float; Target circularity score to dynamically shave from top-down "
                             "(default: 0.5)")
    parser.add_argument("-sb", "--shave_bottom", type=float, default=0.4,
                        help="Float; Target extent score to dynamically shave from bottom-up "
                             "(default: 0.3)")
    parser.add_argument("-sa", "--shave_anterior", type=float, default=0.3,
                        help="Float; Target circularity score to dynamically shave from anterior "
                             "(default: 0.3)")
    parser.add_argument("-sl", "--shave_limit", type=int, default=35,
                        help="Int; Voxel shave cap for dynamic top, bottom, and anterior "
                             "(default: 35)")
    parser.add_argument("-g", "--generate_files", action="store_true",
                        help="Including this flag generates intermediate files (for debugging)")

    args = parser.parse_args()
    auto_segment_lumen(args.input, args.output, args.upper_thresh, args.shave, args.shave_top,
                       args.shave_bottom, args.shave_anterior, args.shave_limit,
                       args.generate_files, args.label, args.tissue_to_air)

if __name__ == "__main__":
    main()