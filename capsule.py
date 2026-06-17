import SimpleITK as sitk
import argparse
import sys
import os
import math
import logging

# --- ALGORITHM CONSTANTS ---
MIN_CAPSULE_VOXELS = 500  # Minimum size to prevent locking onto microscopic dust
SEARCH_RING_RADIUS = 15  # How far to expand the lumen to reach the capsule
CLOSING_RADIUS = 2  # Radius to fuse shattered capsule fragments

# --- LOGGER SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def isolate_capsule(img, lower_thresh, lumen_img, save_debug):
    """
    Segments the prostate capsule using an intensity threshold and intelligently
    locks onto the correct anatomical structure using a 'Centroid Lock' algorithm
    guided by a mandatory pre-segmented lumen mask.
    """
    logger.info(f"Applying lower-bound threshold (>= {lower_thresh})...")
    binary_mask = img >= lower_thresh

    save_debug("1_raw_threshold", sitk.Cast(binary_mask, sitk.sitkUInt8))

    # 1. EXCLUSION ZONE (Lumen is guaranteed to exist)
    logger.info("Applying Lumen exclusion mask to prevent territory overlap...")
    lumen_binary = sitk.Cast(lumen_img > 0, sitk.sitkUInt8)
    safe_zone = sitk.Cast(lumen_img == 0, sitk.sitkUInt8)
    binary_mask = binary_mask * safe_zone

    save_debug("2_after_exclusion", sitk.Cast(binary_mask, sitk.sitkUInt8))

    # --- DEBUG ONLY: Generate a throwaway Label Map to see the shattered state ---
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)
    shattered_components = cc_filter.Execute(binary_mask)
    save_debug("3_shattered_components_BEFORE_healing",
               sitk.Cast(shattered_components, sitk.sitkUInt32))

    # 2. MORPHOLOGICAL CLEANUP
    logger.info(
        f"Applying Morphological Closing (Radius {CLOSING_RADIUS}) to fuse fragmented capsule islands...")
    binary_mask = sitk.BinaryMorphologicalClosing(binary_mask,
                                                  (CLOSING_RADIUS, CLOSING_RADIUS, CLOSING_RADIUS))

    logger.info("Applying Morphological Opening to sweep up static dust...")
    binary_mask = sitk.BinaryMorphologicalOpening(binary_mask, (1, 1, 1))

    # 3. REAL CONNECTED COMPONENTS
    logger.info("Filtering disconnected components from the healed mask...")
    connected_components = cc_filter.Execute(binary_mask)

    save_debug("4_healed_components_AFTER_healing",
               sitk.Cast(connected_components, sitk.sitkUInt32))

    cc_stats = sitk.LabelShapeStatisticsImageFilter()
    cc_stats.Execute(connected_components)

    # 4. INTELLIGENT TARGETING (CENTROID LOCK)
    logger.info(
        f"Using a {SEARCH_RING_RADIUS}-voxel search net and Centroid Lock to find the true capsule...")

    dilated_lumen = sitk.BinaryDilate(lumen_binary,
                                      (SEARCH_RING_RADIUS, SEARCH_RING_RADIUS, SEARCH_RING_RADIUS))
    search_ring = dilated_lumen * safe_zone
    save_debug("5_search_ring", sitk.Cast(search_ring, sitk.sitkUInt8))

    overlap_img = connected_components * sitk.Cast(search_ring, connected_components.GetPixelID())
    save_debug("6_overlap_contacts", sitk.Cast(overlap_img > 0, sitk.sitkUInt8))

    overlap_stats = sitk.LabelShapeStatisticsImageFilter()
    overlap_stats.Execute(overlap_img)

    # Find the center of gravity of the original Lumen mask
    lumen_stats = sitk.LabelShapeStatisticsImageFilter()
    lumen_stats.Execute(lumen_binary)

    if lumen_stats.HasLabel(1):
        target_center = lumen_stats.GetCentroid(1)
    else:
        logger.warning("Provided lumen mask is empty! Falling back to absolute image center.")
        size = img.GetSize()
        target_center = img.TransformIndexToPhysicalPoint(
            [size[0] // 2, size[1] // 2, size[2] // 2])

    best_label = 0
    min_dist = float('inf')

    for label in cc_stats.GetLabels():
        if label == 0: continue

        if cc_stats.GetNumberOfPixels(label) < MIN_CAPSULE_VOXELS:
            continue

        if not overlap_stats.HasLabel(label):
            continue

        # The Lock: Find the component closest to the Lumen's center of gravity
        centroid = cc_stats.GetCentroid(label)
        dist = math.dist(centroid, target_center)

        if dist < min_dist:
            min_dist = dist
            best_label = label

    if best_label == 0:
        logger.warning(
            "No valid capsule found touching the lumen! Falling back to largest component.")
        sorted_components = sitk.RelabelComponent(connected_components)
        target_capsule = sorted_components == 1
    else:
        logger.info(f"  -> Locked onto capsule (Label {best_label}) via Centroid Proximity.")
        target_capsule = connected_components == best_label

    return target_capsule


def auto_segment_capsule(args):
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
    else:
        clean_name = os.path.splitext(base_filename)[0]

    output_path = args.output
    if not output_path:
        if args.combine:
            output_path = os.path.join(input_dir, f"{clean_name}_combined_mask.nii.gz")
        else:
            output_path = os.path.join(input_dir, f"{clean_name}_capsule_mask.nii.gz")

    # --- DEBUG FOLDER SETUP ---
    debug_dir = None
    if args.generate_files:
        debug_dir = os.path.join(input_dir, f"{clean_name}_capsule_debug")
        os.makedirs(debug_dir, exist_ok=True)
        logger.info(f"Debug mode enabled. Intermediate files will be saved to: {debug_dir}")

    def save_debug(key: str, img: sitk.Image):
        if debug_dir:
            out_path = os.path.join(debug_dir, f"{key}.nii.gz")
            logger.info(f"  -> Generating debug file: {out_path}...")
            sitk.WriteImage(img, out_path)

    try:
        logger.info(f"Loading raw scan '{input_path}'...")
        img = sitk.ReadImage(input_path)

        # Lumen is now strictly required
        lumen_path = os.path.abspath(args.lumen_mask)
        if not os.path.exists(lumen_path):
            logger.error(f"The required lumen mask '{lumen_path}' was not found.")
            sys.exit(1)

        logger.info(f"Loading lumen mask '{lumen_path}'...")
        lumen_img = sitk.ReadImage(lumen_path)

        if lumen_img.GetSize() != img.GetSize():
            logger.error(
                "Dimension mismatch! The lumen mask does not match the raw scan dimensions.")
            sys.exit(1)

        logger.info("=== STEP 1: SEGMENT CAPSULE AND CLEAN NOISE ===")
        capsule_mask = isolate_capsule(img, args.threshold, lumen_img, save_debug)

        logger.info("=== STEP 2: FORMAT AND SAVE ===")
        logger.info("Applying final label and realigning metadata...")

        final_mask = sitk.Cast(capsule_mask, sitk.sitkUInt8) * args.label
        final_mask.CopyInformation(img)

        if args.combine:
            logger.info("Combining Capsule and Lumen masks into a single volume...")
            lumen_img_8bit = sitk.Cast(lumen_img, sitk.sitkUInt8)
            final_mask = final_mask + lumen_img_8bit

        logger.info(f"Saving final result to '{output_path}'...")
        sitk.WriteImage(final_mask, output_path)

        logger.info("Processing complete!")

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="High-Contrast Prostate Capsule Segmentation with Sequential Exclusion.")

    parser.add_argument("-i", "--input", required=True,
                        help="Path to raw scan (REQUIRED)")

    # Made the lumen mask REQUIRED since the pipeline now depends on it
    parser.add_argument("-m", "--lumen_mask", required=True,
                        help="Path to the previously generated lumen mask to act as an exclusion zone (REQUIRED)")

    parser.add_argument("-o", "--output", default=None,
                        help="Path to save final mask (Optional: defaults to input directory)")

    parser.add_argument("-c", "--combine", action="store_true",
                        help="Combine the generated capsule mask and the provided lumen mask into one file.")

    parser.add_argument("-l", "--label", type=int, default=1,
                        help="Int; Label number to apply to the capsule (default: 1)")

    parser.add_argument("-t", "--threshold", type=float, default=200.0,
                        help="Float; Lower intensity limit for the capsule (default: 200.0)")

    parser.add_argument("-g", "--generate_files", action="store_true",
                        help="Generate intermediate debug files for troubleshooting.")

    args = parser.parse_args()
    auto_segment_capsule(args)


if __name__ == "__main__":
    main()