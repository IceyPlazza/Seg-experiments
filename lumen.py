import argparse
import SimpleITK as sitk
import numpy as np
import sys
import os
import math

def initial_tissue_mask(img, upper_thresh, gen_file):

    print("Generating initial solid tissue mask...")
    solid_mask = img > upper_thresh
    solid_components = sitk.ConnectedComponent(solid_mask)
    largest_solid = sitk.RelabelComponent(solid_components) == 1

    if gen_file:
        print("[DEBUG] Generating step1_initial_tissue_body.nii.gz...")
        sitk.WriteImage(sitk.Cast(largest_solid, sitk.sitkUInt8),
                        "step1_initial_tissue_body.nii.gz")

    return sitk.GetArrayFromImage(largest_solid)

def extract_middle(solid_array):

    print("Extracting the middle Z-slice as the clean reference blueprint...")
    mid_z = solid_array.shape[0] // 2
    reference_slice = solid_array[mid_z, :, :]

    valid_y, valid_x = np.where(reference_slice)
    if len(valid_y) == 0 or len(valid_x) == 0:
        print("Error: No solid phantom tissue detected on the middle slice.")
        sys.exit(1)

    min_x, max_x = np.min(valid_x), np.max(valid_x)
    min_y, max_y = np.min(valid_y), np.max(valid_y)

    return min_x, max_x, min_y, max_y, reference_slice

def calc_crop(min_x, max_x, min_y, max_y, reference_slice, tissue_to_air):

    print(f"Calculating boundaries using the 4-sided {tissue_to_air} tissue rule...")
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
    print(f"Executing 3D Crop: X({crop_left} to {crop_right}), Y({crop_top} to {crop_bottom})")
    cropped_array = img_array[:, crop_top:crop_bottom + 1, crop_left:crop_right + 1]

    cropped_img = sitk.GetImageFromArray(cropped_array)
    cropped_img.SetSpacing(img.GetSpacing())
    cropped_img.SetDirection(img.GetDirection())
    cropped_img.SetOrigin(img.TransformIndexToPhysicalPoint([crop_left, crop_top, 0]))

    if gen_file:
        print("[DEBUG] Generating step4_cropped_raw_image.nii.gz...")
        sitk.WriteImage(cropped_img, "step4_cropped_raw_image.nii.gz")

    return cropped_img

def segment_air(cropped_img, upper_thresh, gen_file):
    print("Segmenting trapped air within the cropped boundaries...")
    cropped_tissue_mask = cropped_img > upper_thresh
    padded_tissue = sitk.ConstantPad(cropped_tissue_mask, [1, 1, 1], [1, 1, 1], 1)

    filled_tissue = sitk.BinaryFillhole(padded_tissue)
    sealed_shell = sitk.Crop(filled_tissue, [1, 1, 1], [1, 1, 1])

    air_mask = cropped_img <= upper_thresh
    internal_air = air_mask * sealed_shell

    if gen_file:
        print("[DEBUG] Generating step5_cropped_all_internal_air.nii.gz...")
        sitk.WriteImage(sitk.Cast(internal_air, sitk.sitkUInt8),
                        "step5_cropped_all_internal_air.nii.gz")

    return internal_air

def calc_top_shave(internal_air, target_circularity, shave_limit):
    print(f"Analyzing top slices for Circularity (Target >= {target_circularity})...")
    depth = internal_air.GetDepth()
    image_width = internal_air.GetWidth()
    image_height = internal_air.GetHeight()
    shave_count = 0

    # Create the SimpleITK shape analyzer tool
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)

    # Iterate from the top slice downwards
    for z in range(depth -1, -1, -1):

        # Extract the 2D slice
        slice_2d = internal_air[:, :, z]

        # Ensure we only look at the largest air pocket on this slice to avoid noise
        cc_2d = cc_filter.Execute(slice_2d)
        largest_2d_component = sitk.RelabelComponent(cc_2d)

        # Run the shape math
        shape_stats.Execute(largest_2d_component)

        # Guardrail: If the slice is empty, no labels are found
        if not shape_stats.HasLabel(1):
            shave_count += 1
            continue

        bbox = shape_stats.GetBoundingBox(1)
        bbox_width = bbox[2]
        bbox_height = bbox[3]

        # PROPORTION GUARDRAIL: Reject massive background blocks spanning the image edges
        if bbox_width > 0.80 * image_width or bbox_height > 0.80 * image_height:
            shave_count += 1
            continue

        area = shape_stats.GetPhysicalSize(1)
        perimeter = shape_stats.GetPerimeter(1)

        # Guardrail: Prevent division by zero on single-pixel artifacts
        if perimeter == 0:
            shave_count += 1
            continue

        # The Circularity Formula
        circularity = (4 * math.pi * area) / (perimeter ** 2)

        # If the shape is too jagged/messy, it's an artifact. Keep shaving.
        if circularity < target_circularity:
            shave_count += 1
        else:
            # We hit a smooth circle! Stop shaving.
            print(
                f" -> Found valid circular lumen at depth {shave_count} (Circularity: {circularity:.2f})")
            break

    # SAFETY GUARDRAIL

    if shave_count == depth:
        print(" -> WARNING: No top slice met the circularity target. Reverting to 0 shave.")
        return 0
    elif shave_count > shave_limit:
        print(" -> Excessive depth, shave capped at 30.")
        return shave_limit

    return shave_count

def calc_bottom_shave(internal_air, target_extent, shave_limit):
    print(f"Analyzing bottom slices for elliptical Extent (Target >= {target_extent})...")
    depth = internal_air.GetDepth()
    image_width = internal_air.GetWidth()
    image_height = internal_air.GetHeight()
    shave_count = 0

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)

    # Iterate from the absolute bottom slice upwards
    for z in range(0, depth):

        # Extract the 2D slice
        slice_2d = internal_air[:, :, z]

        # Isolate the largest component on this specific slice
        cc_2d = cc_filter.Execute(slice_2d)
        largest_2d = sitk.RelabelComponent(cc_2d)

        shape_stats.Execute(largest_2d)

        if not shape_stats.HasLabel(1):
            shave_count += 1
            continue

        # GetBoundingBox returns [startX, startY, width, height] for 2D images
        bbox = shape_stats.GetBoundingBox(1)
        bbox_width = bbox[2]
        bbox_height = bbox[3]

        # PROPORTION GUARDRAIL: Reject massive background blocks spanning the image edges
        if bbox_width > 0.80 * image_width or bbox_height > 0.80 * image_height:
            shave_count += 1
            continue

        # ASPECT RATIO GUARDRAIL: The Pancake Destroyer
        if bbox_height > 0:
            aspect_ratio = bbox_width / bbox_height
            if aspect_ratio > 4.0:
                shave_count += 1
                continue

        area_pixels = shape_stats.GetNumberOfPixels(1)
        bbox_area = bbox_width * bbox_height

        # Guardrail: Prevent division by zero
        if bbox_area == 0:
            shave_count += 1
            continue

        # The Extent Formula
        extent = area_pixels / bbox_area

        if extent < target_extent:
            shave_count += 1
        else:
            print(
                f" -> Hit valid elliptical lumen. Dropping bottom {shave_count} slices (Score: {extent:.2f})")
            break

    # SAFETY GUARDRAIL
    if shave_count == depth:
        print(" -> WARNING: No bottom slice met the extent target. Reverting to 0 shave.")
        return 0
    elif shave_count > shave_limit:
        print(f" -> Excessive depth, shave capped at {shave_limit}.")
        return shave_limit

    return shave_count

def calc_anterior_shave(internal_air, target_circularity, shave_limit):
    print(f"Analyzing Anterior (-Y) slices for semi-circular shape"
          f" (Target Circularity >= {target_circularity})...")
    image_width = internal_air.GetWidth()
    image_height = internal_air.GetHeight() # This is the Y-axis we are marching through
    image_depth = internal_air.GetDepth()
    shave_count = 0

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)

    # Iterate inward from the -Y edge (Y = 0 moving downward/posteriorly)
    for y in range(0, image_height):

        # Extract the X-Z plane (Coronal view)
        slice_2d = internal_air[:, y, :]

        cc_2d = cc_filter.Execute(slice_2d)
        largest_2d = sitk.RelabelComponent(cc_2d)

        shape_stats.Execute(largest_2d)

        if not shape_stats.HasLabel(1):
            shave_count += 1
            continue

        bbox = shape_stats.GetBoundingBox(1)
        # Because the slice is X-Z, bbox width/height map to original Width/Depth
        bbox_w = bbox[2]
        bbox_h = bbox[3]

        # PROPORTION GUARDRAIL: Reject massive artifact walls
        if bbox_w > 0.80 * image_width or bbox_h > 0.80 * image_depth:
            shave_count += 1
            continue

        area = shape_stats.GetPhysicalSize(1)
        perimeter = shape_stats.GetPerimeter(1)

        if perimeter == 0:
            shave_count += 1
            continue

        # The Circularity Formula
        circularity = (4 * math.pi * area) / (perimeter ** 2)

        if circularity < target_circularity:
            shave_count += 1
        else:
            print(f" -> Found valid semi-circular lumen at Y-depth {shave_count} (Circularity: {circularity:.2f})")
            break

    # SAFETY GUARDRAILS
    if shave_count == image_height:
        print(" -> WARNING: No anterior slice met the target. Reverting to 0 shave.")
        return 0
    elif shave_count > shave_limit:
        print(f" -> Excessive depth. Shave capped at {shave_limit}.")
        return shave_limit

    return shave_count

def shave_faces(internal_air, shave_global, dynamic_shave_top, dynamic_shave_bottom,
                dynamic_shave_anterior, gen_file):
    # --- RESOLVE SHAVE PARAMETERS FOR EACH FACE [X, Y, Z] ---
    # Resolve Z-axis
    z_shave_bottom = dynamic_shave_bottom
    z_shave_top = dynamic_shave_top

    # Resolve X-axis
    x_shave_left = shave_global
    x_shave_right = shave_global

    # Resolve Y-axis
    y_shave_anterior = dynamic_shave_anterior
    y_shave_posterior = shave_global

    # SAFETY GUARDRAIL: Ensure total Z shave doesn't exceed image depth
    if z_shave_bottom + z_shave_top >= internal_air.GetDepth():
        print(" -> WARNING: Combined Z-shaves exceed image depth! Reverting Z-shave to 0.")
        z_shave_bottom = 0
        z_shave_top = 0

    # SAFETY GUARDRAIL: Ensure total Y shave doesn't exceed image height
    if y_shave_anterior + y_shave_posterior >= internal_air.GetHeight():
        print(" -> WARNING: Combined Y-shaves exceed image height! Reverting Y-shave to 0.")
        y_shave_anterior = 0
        y_shave_posterior = 0

    lower_shave_bounds = [x_shave_left, y_shave_anterior, z_shave_bottom]
    upper_shave_bounds = [x_shave_right, y_shave_posterior, z_shave_top]

    print(f"Applying Boundary Erosion -> "
          f"Lower Bounds [X,Y,Z]: {lower_shave_bounds}, Upper Bounds [X,Y,Z]: {upper_shave_bounds}")
    shaved_air = sitk.Crop(internal_air, lower_shave_bounds, upper_shave_bounds)

    # Pad back precisely what was shaved using the asymmetric maps to maintain matrix dimensions
    disconnected_air = sitk.ConstantPad(shaved_air, lower_shave_bounds, upper_shave_bounds, 0)

    print("Filtering disconnected components...")
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)
    air_components = cc_filter.Execute(disconnected_air)
    largest_lumen = sitk.RelabelComponent(air_components) == 1

    if gen_file:
        print("[DEBUG] Generating step6_cropped_isolated_lumen.nii.gz...")
        sitk.WriteImage(sitk.Cast(largest_lumen, sitk.sitkUInt8),
                        "step6_cropped_isolated_lumen.nii.gz")

    return largest_lumen

def auto_segment_lumen(input_path, output_path, upper_thresh, shave_global, target_top,
                       target_bottom, target_anterior, shave_limit, gen_file, target_label, tissue_to_air):
    if not os.path.exists(input_path):
        print(f"\nError: The input scan '{input_path}' was not found.")
        sys.exit(1)

    try:
        print(f"\nLoading raw scan '{input_path}'...")
        img = sitk.ReadImage(input_path)
        img_array = sitk.GetArrayFromImage(img)

        print("\n --- STEP 1: INITIAL TISSUE MASK --- ")
        solid_array = initial_tissue_mask(img, upper_thresh, gen_file)

        print("\n --- STEP 2: THE 'GOLDEN SLICE' REFERENCE ---")
        min_x, max_x, min_y, max_y, reference_slice = extract_middle(solid_array)

        print("\n --- STEP 3: CALCULATE 4-WAY CROP LINES ---")
        crop_top, crop_bottom, crop_left, crop_right = (
            calc_crop(min_x, max_x, min_y, max_y, reference_slice, tissue_to_air))

        print("\n --- STEP 4: EXECUTE THE 3D CROP --- ")
        cropped_img = perform_crop(img, img_array, crop_top, crop_bottom, crop_left, crop_right,
                                   gen_file)

        print("\n --- STEP 5: SEGMENT INTERNAL AIR --- ")
        internal_air = segment_air(cropped_img, upper_thresh, gen_file)

        print("\n --- STEP 6: ASYMMETRICALLY SHAVE SIDES --- ")
        dynamic_shave_top = calc_top_shave(internal_air, target_top, shave_limit)
        dynamic_shave_bottom = calc_bottom_shave(internal_air, target_bottom, shave_limit)
        dynamic_shave_anterior = calc_anterior_shave(internal_air, target_anterior, shave_limit)
        shaved_lumen = shave_faces(internal_air, shave_global, dynamic_shave_top,
                                    dynamic_shave_bottom, dynamic_shave_anterior, gen_file)

        print("\n --- STEP 7: PASTE BACK TO ORIGINAL DIMENSIONS --- ")
        print("Realigning mask with original dimensions...")
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

        print(f"\nSaving final result to '{output_path}'...")
        sitk.WriteImage(final_full_mask, output_path)
        print("Processing complete!\n")

    except Exception as e:
        print(f"An error occurred during processing: {e}\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive BPH Lumen Segmentation.")

    # Required input output paths
    parser.add_argument("-i", "--input", required=True,
                        help="Path to raw scan (REQUIRED)")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to save final mask (REQUIRED)")

    # Which label to apply mask to
    parser.add_argument("-l", "--label", type=int, default=2,
                        help="Int; Label number to work on (default: 2)")

    # Adjust upper threshold
    parser.add_argument("-u", "--upper_thresh", type=float, default=-500.0,
                        help="Float; Upper threshold limit (default: -500)")

    # Tissue-to-air ratio requirement
    parser.add_argument("-t", "--tissue_to_air", type=float, default=0.92,
                        help="Float; Tissue-to-air ratio limit (default: 0.92)")

    # Global shave parameters
    parser.add_argument("-s", "--shave", type=int, default=5,
                        help="Int; Global number of voxels to shave off all 6 faces (default: 5)")

    # Specialized face overrides
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

    # Generate intermediate files (for debugging)
    parser.add_argument("-g", "--generate_files", action="store_true",
                        help="Including this flag generates intermediate files (for debugging)")

    args = parser.parse_args()
    auto_segment_lumen(args.input, args.output, args.upper_thresh, args.shave, args.shave_top,
                       args.shave_bottom, args.shave_anterior, args.shave_limit,
                       args.generate_files, args.label, args.tissue_to_air)


if __name__ == "__main__":
    main()