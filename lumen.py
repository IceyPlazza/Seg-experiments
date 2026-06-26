import argparse
import SimpleITK as sitk
import numpy as np
import sys
import os
import math
import logging

# --- ALGORITHM CONSTANTS ---
MAX_BBOX_PROPORTION = 0.80  # Max bounding box size before rejecting as artifact wall
MAX_ASPECT_RATIO = 4.0  # Max width-to-height ratio to reject stretched "pancake" artifacts
CROP_BOTTOM_OFFSET = 0.03  # Tissue-to-air ratio offset for the posterior/bottom crop line
CROP_TOP_OFFSET = 0.03  # Tissue-to-air ratio offset for the anterior/top crop line

# --- LOGGER SETUP ---
logger = logging.getLogger(__name__)


def initial_tissue_mask(img, upper_thresh, save_debug):
    """
    Generates a binary mask of the largest solid tissue component in the image.

    Args:
        img (SimpleITK.Image): The original 3D image scan.
        upper_thresh (float): The threshold value separating solid tissue from air.
        save_debug (callable): Function to save intermediate debug steps.

    Returns:
        numpy.ndarray: A 3D numpy array representing the largest solid tissue mask.
    """
    logger.info("Generating initial solid tissue mask...")
    solid_mask = img > upper_thresh
    solid_components = sitk.ConnectedComponent(solid_mask)
    largest_solid = sitk.RelabelComponent(solid_components) == 1

    save_debug("initial_tissue_body", sitk.Cast(largest_solid, sitk.sitkUInt8))
    return sitk.GetArrayFromImage(largest_solid)


def extract_middle(solid_array):
    """
    Extracts the middle Z-slice from the solid mask to use as a 2D reference blueprint.

    Args:
        solid_array (numpy.ndarray): The 3D solid tissue mask array.

    Returns:
        tuple: A tuple containing (min_x, max_x, min_y, max_y, reference_slice),
               defining the bounding box coordinates and the 2D array of the middle slice.
    """
    logger.info("Extracting the middle Z-slice as the clean reference blueprint...")
    mid_z = solid_array.shape[0] // 2
    reference_slice = solid_array[mid_z, :, :]

    valid_y, valid_x = np.where(reference_slice)
    if len(valid_y) == 0 or len(valid_x) == 0:
        logger.error("No solid phantom tissue detected on the middle slice.")
        sys.exit(1)

    min_x, max_x = np.min(valid_x), np.max(valid_x)
    min_y, max_y = np.min(valid_y), np.max(valid_y)

    return min_x, max_x, min_y, max_y, reference_slice


def calc_crop(min_x, max_x, min_y, max_y, reference_slice, tissue_to_air):
    """
    Calculates tight 2D crop boundaries based on a tissue-to-air density ratio.

    Iterates inward from the bounding box edges of the reference slice until
    the line's tissue-to-air density exceeds the required thresholds.

    Args:
        min_x (int): Minimum X coordinate from the reference bounding box.
        max_x (int): Maximum X coordinate from the reference bounding box.
        min_y (int): Minimum Y coordinate from the reference bounding box.
        max_y (int): Maximum Y coordinate from the reference bounding box.
        reference_slice (numpy.ndarray): The 2D middle slice used for calculations.
        tissue_to_air (float): The base required ratio of tissue to air to trigger a crop.

    Returns:
        tuple: A tuple containing the calculated coordinates (crop_top, crop_bottom, crop_left,
        crop_right).
    """
    logger.info(f"Calculating boundaries using the 4-sided {tissue_to_air} tissue rule...")
    crop_bottom = int(max_y)
    for y in range(max_y, min_y - 1, -1):
        row_segment = reference_slice[y, min_x:max_x + 1]
        tissue_ratio = np.sum(row_segment) / len(row_segment)
        if tissue_ratio > tissue_to_air + CROP_BOTTOM_OFFSET:
            crop_bottom = int(y)
            break

    crop_top = int(min_y)
    for y in range(min_y, crop_bottom + 1):
        row_segment = reference_slice[y, min_x:max_x + 1]
        tissue_ratio = np.sum(row_segment) / len(row_segment)
        if tissue_ratio > tissue_to_air - CROP_TOP_OFFSET:
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


def perform_crop(img, img_array, crop_top, crop_bottom, crop_left, crop_right, save_debug):
    """
    Crops the original 3D image using the computed 2D boundaries.

    Args:
        img (SimpleITK.Image): The original 3D scan with physical metadata.
        img_array (numpy.ndarray): The raw data array of the original scan.
        crop_top (int): Calculated top boundary (Y-axis).
        crop_bottom (int): Calculated bottom boundary (Y-axis).
        crop_left (int): Calculated left boundary (X-axis).
        crop_right (int): Calculated right boundary (X-axis).
        save_debug (callable): Function to save intermediate debug steps.

    Returns:
        SimpleITK.Image: The cropped 3D image with preserved spatial metadata.
    """
    logger.info(
        f"Executing 2D Crop: X({crop_left} to {crop_right}), Y({crop_top} to {crop_bottom})")
    cropped_array = img_array[:, crop_top:crop_bottom + 1, crop_left:crop_right + 1]

    cropped_img = sitk.GetImageFromArray(cropped_array)
    cropped_img.SetSpacing(img.GetSpacing())
    cropped_img.SetDirection(img.GetDirection())
    cropped_img.SetOrigin(img.TransformIndexToPhysicalPoint([crop_left, crop_top, 0]))

    save_debug("cropped_raw_image", cropped_img)
    return cropped_img


def segment_air(cropped_img, upper_thresh, save_debug):
    """
    Isolates trapped internal air pockets within the cropped tissue volume.

    Achieves isolation by creating a watertight padded shell of the tissue mask,
    filling internal holes, and intersecting it with the low-density image regions.

    Args:
        cropped_img (SimpleITK.Image): The cropped 3D image.
        upper_thresh (float): Threshold to differentiate air from tissue.
        save_debug (callable): Function to save intermediate debug steps.

    Returns:
        SimpleITK.Image: A 3D binary mask isolating the internal air.
    """
    logger.info("Segmenting trapped air within the cropped boundaries...")
    cropped_tissue_mask = cropped_img > upper_thresh
    padded_tissue = sitk.ConstantPad(cropped_tissue_mask, [1, 1, 1], [1, 1, 1], 1)

    filled_tissue = sitk.BinaryFillhole(padded_tissue)
    sealed_shell = sitk.Crop(filled_tissue, [1, 1, 1], [1, 1, 1])

    air_mask = cropped_img <= upper_thresh
    internal_air = air_mask * sealed_shell

    save_debug("cropped_all_internal_air", sitk.Cast(internal_air, sitk.sitkUInt8))
    return internal_air


def calc_top_shave(internal_air, target_extent, shave_limit):
    """
    Dynamically calculates the number of slices to erode from the Superior (Top / S) face.

    Marches downward from the top slice (Z = depth - 1), isolating the largest 2D air
    component on each Axial plane. It evaluates the 'Extent' score (Area / BoundingBox Area)
    to bypass irregular artifacts. Shaving stops when the cross-section solidifies into a
    clean, measurable structure that meets the target Extent score.

    Includes an Aspect Ratio guardrail to actively destroy horizontal "pancake" artifacts
    that artificially mimic high Extent scores.

    Args:
        internal_air (SimpleITK.Image): 3D binary mask of internal air pockets.
        target_extent (float): The minimum Extent score required to stop shaving.
        shave_limit (int): The maximum allowable slices to erode before triggering a safety abort.

    Returns:
        int: The precise number of slices to safely shave from the Superior face.
    """
    logger.info(f"Analyzing Superior (S/Top) slices for clean Extent (Target >= {target_extent})...")
    depth = internal_air.GetDepth()
    image_width = internal_air.GetWidth()
    image_height = internal_air.GetHeight()
    shave_count = 0

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)

    for z in range(depth - 1, -1, -1):
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

        if (bbox_width > MAX_BBOX_PROPORTION * image_width
                or bbox_height > MAX_BBOX_PROPORTION * image_height):
            shave_count += 1
            continue

        # Aspect ratio check (The Pancake Destroyer)
        if bbox_height > 0:
            aspect_ratio = bbox_width / bbox_height
            if aspect_ratio > MAX_ASPECT_RATIO:
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
            logger.info(
                f"  -> [SUCCESS] Hit valid elliptical lumen. "
                f"Dropping top {shave_count} slices (Score: {extent:.2f})")
            break

    if shave_count == depth:
        logger.warning("  -> No top slice met the extent target. Reverting to 0 shave.")
        return 0
    elif shave_count > shave_limit:
        logger.warning(f"  -> Excessive depth, shave capped at {shave_limit}.")
        return shave_limit

    return shave_count


def calc_bottom_shave(internal_air, target_circularity, shave_limit):
    """
    Dynamically calculates the number of slices to erode from the Inferior (Bottom / I) face.

    Marches upward from the absolute bottom slice (Z = 0), isolating the largest 2D air
    component on each Axial plane. It evaluates the 'Circularity' score (4π * Area / Perimeter²)
    to bypass messy, highly-perimetered artifact walls. Shaving stops when the algorithm
    detects the smooth, tubular entry point of the true lumen.

    Args:
        internal_air (SimpleITK.Image): 3D binary mask of internal air pockets.
        target_circularity (float): The minimum Circularity score required to stop shaving.
        shave_limit (int): The maximum allowable slices to erode before triggering a safety abort.

    Returns:
        int: The precise number of slices to safely shave from the Inferior face.
    """
    logger.info(f"Analyzing Inferior (I/Bottom) slices for Circularity (Target >= {target_circularity})...")
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

        if (bbox_width > MAX_BBOX_PROPORTION * image_width
                or bbox_height > MAX_BBOX_PROPORTION * image_height):
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
            logger.info(
                f"  -> [SUCCESS] Found valid circular lumen at depth {shave_count} "
                f"(Circularity: {circularity:.2f})")
            break

    if shave_count == depth:
        logger.warning("  -> No bottom slice met the circularity target. Reverting to 0 shave.")
        return 0
    elif shave_count > shave_limit:
        logger.warning(f"  -> Excessive depth, shave capped at {shave_limit}.")
        return shave_limit

    return shave_count


def calc_anterior_shave(internal_air, target_extent, shave_limit, z_shave_top, z_shave_bottom):
    """
    Dynamically calculates the number of slices to erode from the Anterior (Front / -Y) face.

    Extracts Coronal (X-Z) slices marching inward from Y = 0. To prevent the "T-Shape Trap"
    (where horizontal floor/ceiling artifacts fuse with the vertical true lumen), this
    function sequentially applies the pre-calculated Z-axis limits before performing
    its 2D shape analysis.

    Evaluates the Extent score to lock onto the rectangular bounding box of the vertical tube.
    Explicitly ignores Z-height proportion constraints to naturally accommodate the tall anatomy
    of the lumen body.

    Args:
        internal_air (SimpleITK.Image): 3D binary mask of internal air pockets.
        target_extent (float): Minimum Extent score required for Coronal slice validation.
        shave_limit (int): Maximum allowable slices to erode before triggering a safety abort.
        z_shave_top (int): Previously calculated Superior shave limit (used for isolation).
        z_shave_bottom (int): Previously calculated Inferior shave limit (used for isolation).

    Returns:
        int: The precise number of slices to safely shave from the Anterior face.
    """
    logger.info(
        f"Analyzing Anterior (-Y) slices for semi-elliptical shape (Target >= {target_extent})...")
    image_width = internal_air.GetWidth()
    image_height = internal_air.GetHeight()
    image_depth = internal_air.GetDepth()
    shave_count = 0

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)

    # APPLY SEQUENTIAL LIMITS: Ignore the Z-axis artifacts that we already mapped!
    z_start = z_shave_bottom
    z_end = image_depth - z_shave_top

    for y in range(0, image_height):
        # Extract the X-Z plane, ignoring the polluted top/bottom slices
        slice_2d = internal_air[:, y, z_start:z_end]

        cc_2d = cc_filter.Execute(slice_2d)
        largest_2d = sitk.RelabelComponent(cc_2d)
        shape_stats.Execute(largest_2d)

        if not shape_stats.HasLabel(1):
            shave_count += 1
            continue

        bbox = shape_stats.GetBoundingBox(1)
        bbox_w = bbox[2]
        bbox_h = bbox[3]

        if bbox_w > MAX_BBOX_PROPORTION * image_width:
            shave_count += 1
            continue

        if bbox_h > 0:
            aspect_ratio = bbox_w / bbox_h
            if aspect_ratio > MAX_ASPECT_RATIO:
                shave_count += 1
                continue

        area_pixels = shape_stats.GetNumberOfPixels(1)
        bbox_area = bbox_w * bbox_h

        if bbox_area == 0:
            shave_count += 1
            continue

        extent = area_pixels / bbox_area

        if extent < target_extent:
            shave_count += 1
        else:
            logger.info(
                f"  -> [SUCCESS] Found valid semi-circular lumen at Y-depth {shave_count} "
                f"(Extent: {extent:.2f})")
            break

    if shave_count == image_height:
        logger.warning("  -> No anterior slice met the target. Reverting to 0 shave.")
        return 0
    elif shave_count > shave_limit:
        logger.warning(f"  -> Excessive depth. Shave capped at {shave_limit}.")
        return shave_limit

    return shave_count


def shave_faces(internal_air, shave_global, dynamic_shave_top, dynamic_shave_bottom,
                dynamic_shave_anterior, save_debug):
    """
    Executes boundary erosion using the calculated asymmetric limits.

    Crops the image according to the dynamic values, pads it back to its original
    cropped volume to maintain matrix dimensions, and extracts the largest contiguous component.

    Args:
        internal_air (SimpleITK.Image): Binary mask of internal air pockets.
        shave_global (int): Base uniform voxel shave applied to lateral/posterior faces.
        dynamic_shave_top (int): Slices to erode from the top face.
        dynamic_shave_bottom (int): Slices to erode from the bottom face.
        dynamic_shave_anterior (int): Slices to erode from the anterior face.
        save_debug (callable): Function to save intermediate debug steps.

    Returns:
        SimpleITK.Image: A 3D mask containing the final isolated continuous lumen.
    """
    z_shave_bottom = dynamic_shave_bottom
    z_shave_top = dynamic_shave_top
    x_shave_left = shave_global
    x_shave_right = shave_global
    y_shave_anterior = dynamic_shave_anterior
    y_shave_posterior = shave_global

    if z_shave_bottom + z_shave_top >= internal_air.GetDepth():
        logger.warning("  -> Combined Z-shaves exceed image depth! Reverting Z-shave to 0.")
        z_shave_bottom = 0
        z_shave_top = 0

    if y_shave_anterior + y_shave_posterior >= internal_air.GetHeight():
        logger.warning("  -> Combined Y-shaves exceed image height! Reverting Y-shave to 0.")
        y_shave_anterior = 0
        y_shave_posterior = 0

    lower_shave_bounds = [x_shave_left, y_shave_anterior, z_shave_bottom]
    upper_shave_bounds = [x_shave_right, y_shave_posterior, z_shave_top]

    logger.info(
        f"Applying Boundary Erosion "
        f"-> Lower [X,Y,Z]: {lower_shave_bounds}, Upper [X,Y,Z]: {upper_shave_bounds}")
    shaved_air = sitk.Crop(internal_air, lower_shave_bounds, upper_shave_bounds)

    disconnected_air = sitk.ConstantPad(shaved_air, lower_shave_bounds, upper_shave_bounds, 0)

    logger.info("Filtering disconnected components...")
    disconnected_air = sitk.BinaryMorphologicalOpening(disconnected_air, (1,1,1))
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)
    air_components = cc_filter.Execute(disconnected_air)
    largest_lumen = sitk.RelabelComponent(air_components) == 1

    save_debug("cropped_isolated_lumen", sitk.Cast(largest_lumen, sitk.sitkUInt8))
    return largest_lumen


def auto_segment_lumen(args):
    """
    Orchestrates the 7-step automated pipeline for BPH lumen segmentation.

    Reads the raw image, performs tissue extraction, identifies tight crop boundaries,
    isolates internal air, executes asymmetric morphological shaving based on dynamic
    geometric constraints, and repastes the extracted lumen back into the original spatial frame.

    Args:
        args (argparse.Namespace): The parsed command-line arguments containing all configurations
                                   (input path, output constraints, targets, threshold flags).
    """
    input_path = os.path.abspath(args.input)

    if not os.path.exists(input_path):
        logger.error(f"The input scan '{input_path}' was not found.")
        sys.exit(1)

    input_dir = os.path.dirname(input_path)
    base_filename = os.path.basename(input_path)

    if base_filename.endswith('_0000.nii.gz'):
        clean_name = base_filename[:-12]
    elif base_filename.endswith('.nii.gz'):
        clean_name = base_filename[:-7]
    elif base_filename.endswith('.nii'):
        clean_name = base_filename[:-4]
    else:
        clean_name = os.path.splitext(base_filename)[0]

    output_path = os.path.join(input_dir, f"{clean_name}_lumen_mask.nii.gz")

    debug_dir = None
    if args.generate_files:
        debug_dir = os.path.join(input_dir, f"{clean_name}_debug")
        os.makedirs(debug_dir, exist_ok=True)
        logger.info(f"Debug mode enabled. Intermediate files will be saved to: {debug_dir}")

    step_counter = 0

    def save_debug(key: str, img: sitk.Image):
        nonlocal step_counter
        if debug_dir:
            step_counter += 1
            out_path = os.path.join(debug_dir, f"step{step_counter}_{key}.nii.gz")
            logger.debug(f"  -> Generating {out_path}...")
            sitk.WriteImage(img, out_path)

    try:
        logger.info(f"Loading raw scan '{input_path}'...")
        img = sitk.ReadImage(input_path)
        img_array = sitk.GetArrayFromImage(img)

        logger.info("=== STEP 1: INITIAL TISSUE MASK ===")
        solid_array = initial_tissue_mask(img, args.upper_thresh, save_debug)

        logger.info("=== STEP 2: THE 'GOLDEN SLICE' REFERENCE ===")
        min_x, max_x, min_y, max_y, reference_slice = extract_middle(solid_array)
        step_counter += 1

        logger.info("=== STEP 3: CALCULATE 4-WAY CROP LINES ===")
        crop_top, crop_bottom, crop_left, crop_right = (
            calc_crop(min_x, max_x, min_y, max_y, reference_slice, args.tissue_to_air))
        step_counter += 1

        logger.info("=== STEP 4: EXECUTE THE 3D CROP ===")
        cropped_img = perform_crop(img, img_array, crop_top, crop_bottom, crop_left, crop_right,
                                   save_debug)

        logger.info("=== STEP 5: SEGMENT INTERNAL AIR ===")
        internal_air = segment_air(cropped_img, args.upper_thresh, save_debug)

        logger.info("=== STEP 6: ASYMMETRICALLY SHAVE SIDES ===")
        dynamic_shave_top = calc_top_shave(internal_air, args.shave_top, args.shave_limit)
        dynamic_shave_bottom = calc_bottom_shave(internal_air, args.shave_bottom, args.shave_limit)
        dynamic_shave_anterior = calc_anterior_shave(internal_air, args.shave_anterior,
                                                     args.shave_limit, dynamic_shave_top,
                                                     dynamic_shave_bottom)

        shaved_lumen = shave_faces(internal_air, args.shave, dynamic_shave_top,
                                   dynamic_shave_bottom, dynamic_shave_anterior, save_debug)

        logger.info("=== STEP 7: PASTE BACK TO ORIGINAL DIMENSIONS ===")
        logger.info("Realigning mask with original dimensions...")
        final_full_mask = sitk.Image(img.GetSize(), sitk.sitkUInt8)
        final_full_mask.CopyInformation(img)

        lumen_mask_8bit = sitk.Cast(shaved_lumen, sitk.sitkUInt8) * args.label
        final_full_mask = sitk.Paste(
            destinationImage=final_full_mask,
            sourceImage=lumen_mask_8bit,
            sourceSize=lumen_mask_8bit.GetSize(),
            sourceIndex=[0, 0, 0],
            destinationIndex=[crop_left, crop_top, 0]
        )

        logger.info(f"Saving final result to '{output_path}'...")
        print(f"\n[INFO] Saving final result to '{output_path}'...")
        sitk.WriteImage(final_full_mask, output_path)
        print("[INFO] Processing complete!\n")

        logger.info("Processing complete!")

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive BPH Lumen Segmentation.")

    parser.add_argument("-i", "--input", required=True,
                        help="Path to raw scan (REQUIRED)")
    parser.add_argument("-l", "--label", type=int, default=2,
                        help="Int; Label number to work on (default: 2)")
    parser.add_argument("-u", "--upper_thresh", type=float, default=-500.0,
                        help="Float; Upper threshold limit (default: -500)")
    parser.add_argument("-t", "--tissue_to_air", type=float, default=0.92,
                        help="Float; Tissue-to-air ratio limit (default: 0.92)")
    parser.add_argument("-s", "--shave", type=int, default=5,
                        help="Int; Global number of voxels to shave off all 6 faces (default: 5)")
    parser.add_argument("-st", "--shave_top", type=float, default=0.3,
                        help="Float; Target extent score to dynamically shave from Superior/Top "
                             "(default: 0.3)")
    parser.add_argument("-sb", "--shave_bottom", type=float, default=0.4,
                        help="Float; Target circularity score to dynamically shave from "
                             "Inferior/Bottom (default: 0.4)")
    parser.add_argument("-sa", "--shave_anterior", type=float, default=0.4,
                        help="Float; Target extent score to dynamically shave from anterior "
                             "(default: 0.4)")
    parser.add_argument("-sl", "--shave_limit", type=int, default=35,
                        help="Int; Voxel shave cap for dynamic top, bottom, and anterior "
                             "(default: 35)")
    parser.add_argument("-g", "--generate_files", action="store_true",
                        help="Including this flag generates intermediate files (for debugging)")
    parser.add_argument("--stl", action="store_true",
                        help="Including this flag generates 3D printable STL meshes of the final labels")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    auto_segment_lumen(args)


if __name__ == "__main__":
    main()