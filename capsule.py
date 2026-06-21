import SimpleITK as sitk
import argparse
import sys
import math
import logging
from pathlib import Path
from typing import Callable, Tuple

# --- LOGGER SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# =====================================================================
# PIPELINE STAGE 1: CORE ALGORITHM METHODS
# =====================================================================

def raw_exclusion(
    img: sitk.Image,
    lower_thresh: float,
    lumen_img: sitk.Image,
    save_debug: Callable
) -> Tuple[sitk.Image, sitk.Image, sitk.Image]:
    """
    Applies an initial lower-bound threshold and carves out the lumen exclusion zone.

    Args:
        img: The raw input medical scan.
        lower_thresh: The minimum intensity value for the capsule.
        lumen_img: The binary or labeled mask representing the lumen.
        save_debug: Callback function to save intermediate files if debug mode is active.

    Returns:
        A tuple containing:
            - binary_mask (sitk.Image): The thresholded image excluding the lumen.
            - lumen_binary (sitk.Image): The binarized lumen mask.
            - safe_zone (sitk.Image): The inverted lumen mask (safe territory).
    """
    logger.info("=== STEP 1: Raw Thresholding & Exclusion ===")
    logger.info(f"Applying lower-bound threshold (>= {lower_thresh})...")
    binary_mask = img >= lower_thresh

    save_debug("1_raw_threshold", sitk.Cast(binary_mask, sitk.sitkUInt8))

    logger.info("Applying Lumen exclusion mask to prevent territory overlap...")
    lumen_binary = sitk.Cast(lumen_img > 0, sitk.sitkUInt8)
    safe_zone = sitk.Cast(lumen_img == 0, sitk.sitkUInt8)

    # Exclude lumen territory
    binary_mask = binary_mask * safe_zone

    return binary_mask, lumen_binary, safe_zone


def extract_components(
    binary_mask: sitk.Image,
    max_labels: int,
    save_debug: Callable
) -> Tuple[sitk.Image, sitk.LabelShapeStatisticsImageFilter]:
    """
    Isolates and measures the largest connected components, filtering out microscopic noise.

    Args:
        binary_mask: The excluded binary mask from Step 1.
        max_labels: The maximum number of top largest components to retain.
        save_debug: Callback function to save intermediate files.

    Returns:
        A tuple containing:
            - final_components (sitk.Image): Labeled image of the top N components.
            - cc_stats (sitk.LabelShapeStatisticsImageFilter): Statistics computed on the components.
    """
    logger.info("=== STEP 2: Component Extraction & Dust Filter ===")

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)
    raw_components = cc_filter.Execute(binary_mask)

    # Sort by size
    final_components_raw = sitk.RelabelComponent(raw_components)
    del raw_components  # RAM flush

    # Isolate ONLY the Top N largest structures to permanently delete dust
    final_components = sitk.Threshold(final_components_raw, lower=1, upper=max_labels, outsideValue=0)
    del final_components_raw  # RAM flush

    save_debug("2_filtered_components", final_components)

    # Calculate statistics only for the surviving top components
    cc_stats = sitk.LabelShapeStatisticsImageFilter()
    cc_stats.Execute(final_components)

    return final_components, cc_stats


def centroid_lock(
    img: sitk.Image,
    final_components: sitk.Image,
    cc_stats: sitk.LabelShapeStatisticsImageFilter,
    lumen_binary: sitk.Image,
    safe_zone: sitk.Image,
    min_voxels: int,
    search_radius: int,
    max_labels: int,
    save_debug: Callable
) -> sitk.Image:
    """
    Identifies the true capsule using a dilated search ring and centroid proximity to the lumen.

    Args:
        img: The raw input medical scan (used for physical coordinate conversion).
        final_components: Labeled image of the top connected components.
        cc_stats: Statistics filter containing data for the final_components.
        lumen_binary: The binarized lumen mask.
        safe_zone: The non-lumen territory mask.
        min_voxels: Minimum required size for a component to be considered.
        search_radius: Voxel radius to dilate the lumen to find contacting structures.
        max_labels: The number of top labels being evaluated.
        save_debug: Callback function to save intermediate files.

    Returns:
        sitk.Image: A binary mask of the successfully locked target capsule.
    """
    logger.info("=== STEP 3: The Centroid Lock (Intelligent Targeting) ===")

    logger.info(f"Using a {search_radius}-voxel search net...")
    dilated_lumen = sitk.BinaryDilate(lumen_binary, (search_radius, search_radius, search_radius))
    search_ring = dilated_lumen * safe_zone
    del dilated_lumen  # RAM flush

    save_debug("3_search_ring", search_ring)

    # Use sitk.Mask to avoid memory blowout
    overlap_img = sitk.Mask(final_components, search_ring)
    del search_ring  # RAM flush

    save_debug("4_overlap_contacts", sitk.Cast(overlap_img > 0, sitk.sitkUInt8))

    overlap_stats = sitk.LabelShapeStatisticsImageFilter()
    overlap_stats.Execute(overlap_img)
    del overlap_img  # RAM flush

    # Calculate lumen center of gravity
    lumen_stats = sitk.LabelShapeStatisticsImageFilter()
    lumen_stats.Execute(lumen_binary)

    lumen_labels = lumen_stats.GetLabels()
    if lumen_labels:
        # Safely grab the first available label
        target_center = lumen_stats.GetCentroid(lumen_labels[0])
    else:
        logger.warning("Provided lumen mask is empty! ...")
        size = img.GetSize()
        target_center = img.TransformIndexToPhysicalPoint(
            [size[0] // 2, size[1] // 2, size[2] // 2])

    best_label = 0
    min_dist = float('inf')

    # Loop through the surviving labels
    top_labels = [l for l in cc_stats.GetLabels() if 0 < l <= max_labels]

    for label in top_labels:
        if cc_stats.GetNumberOfPixels(label) < min_voxels:
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
        target_capsule = final_components == 1
    else:
        logger.info(f"  -> Locked onto capsule (Label {best_label}) via Centroid Proximity.")
        target_capsule = final_components == best_label

    return target_capsule


def distance_map_closing(
    target_capsule: sitk.Image,
    patch_radius: int,
    save_debug: Callable
) -> sitk.Image:
    """
    Applies streamlined distance map patching to close large topological gaps.

    Uses Signed Maurer Distance Maps for memory-efficient hole bridging, followed
    by localized morphological closing to clean surface artifacts.

    Args:
        target_capsule: The binary mask of the isolated capsule.
        patch_radius: The radius (in voxels) to use for distance map bridging.
        save_debug: Callback function to save intermediate files.

    Returns:
        sitk.Image: The fully patched and smoothed binary capsule mask.
    """
    if patch_radius <= 0:
        logger.info("=== STEP 4: Distance Map Patching [SKIPPED] ===")
        return target_capsule

    logger.info("=== STEP 4: Streamlined Distance Map Patching ===")
    save_debug("4b_base_target_capsule", sitk.Cast(target_capsule, sitk.sitkUInt8))

    # Phase 1-3: Maurer Distance Map (Bridging Massive Holes)
    logger.info(f"Applying Distance-Based massive bridge (Radius {patch_radius})...")
    distance_filter = sitk.SignedMaurerDistanceMapImageFilter()
    distance_filter.SetUseImageSpacing(False)
    distance_filter.SetSquaredDistance(False)

    logger.info(f"  -> Performing dilution...")
    dist_map = distance_filter.Execute(target_capsule)
    dilated_capsule = dist_map <= patch_radius
    del dist_map  # [RAM CLEAR]
    save_debug("5a_dist_dilated", sitk.Cast(dilated_capsule, sitk.sitkUInt8))

    dist_map = distance_filter.Execute(dilated_capsule)
    del dilated_capsule  # [RAM CLEAR]

    logger.info(f"  -> Performing erosion...")
    eroded_capsule = dist_map < -patch_radius
    del dist_map  # [RAM CLEAR]
    save_debug("5b_dist_eroded", sitk.Cast(eroded_capsule, sitk.sitkUInt8))

    logger.info(f"  -> Bridging capsule...")
    bridged_capsule = eroded_capsule | target_capsule
    del eroded_capsule  # [RAM CLEAR]
    save_debug("5c_dist_bridged", sitk.Cast(bridged_capsule, sitk.sitkUInt8))

    # Phase 4: Anti-Aliasing & Surface Cleanup
    logger.info("Applying localized morphological closing for surface cleanup...")
    final_capsule = sitk.BinaryMorphologicalClosing(bridged_capsule, (4, 4, 4))
    save_debug("5d_final_patched_shell", sitk.Cast(final_capsule, sitk.sitkUInt8))

    return final_capsule

# =====================================================================
# PIPELINE STAGE 2: ORCHESTRATION & EXPORT
# =====================================================================

def isolate_capsule(
    img: sitk.Image,
    lower_thresh: float,
    lumen_img: sitk.Image,
    min_voxels: int,
    search_radius: int,
    max_labels: int,
    patch_radius: int,
    save_debug: Callable
) -> sitk.Image:
    """
    Orchestrates the 4-step algorithm to isolate the prostate capsule.

    Args:
        img: Raw input scan.
        lower_thresh: Minimum intensity threshold.
        lumen_img: Lumen exclusion mask.
        min_voxels: Minimum component size.
        search_radius: Lumen dilation search radius.
        max_labels: Maximum components to evaluate.
        patch_radius: Distance map closing radius.
        save_debug: Debug file generator.

    Returns:
        sitk.Image: The fully isolated and processed capsule binary mask.
    """
    binary_mask, lumen_binary, safe_zone = raw_exclusion(
        img, lower_thresh, lumen_img, save_debug)

    final_components, cc_stats = extract_components(
        binary_mask, max_labels, save_debug)
    del binary_mask  # [RAM CLEAR]

    target_capsule = centroid_lock(
        img, final_components, cc_stats, lumen_binary, safe_zone,
        min_voxels, search_radius, max_labels, save_debug
    )
    del final_components, cc_stats, safe_zone  # [RAM CLEAR]

    patched_capsule = distance_map_closing(
        target_capsule, patch_radius, save_debug)

    del target_capsule, lumen_binary

    return patched_capsule


def auto_segment_capsule(args: argparse.Namespace) -> None:
    """
    Validates inputs, initializes the debug environment, runs the isolation
    pipeline, and handles standard or combined IO export operations.
    """
    input_path = Path(args.input).resolve()

    if not input_path.exists():
        logger.error(f"The input scan '{input_path}' was not found.")
        raise FileNotFoundError(f"Missing input scan: {input_path}")

    input_dir = input_path.parent
    filename = input_path.name

    # Clean up standard medical imaging suffixes
    if filename.endswith('_0000.nii.gz'):
        clean_name = filename[:-12]
    elif filename.endswith('.nii.gz'):
        clean_name = filename[:-7]
    else:
        clean_name = input_path.stem

    # --- DEBUG FOLDER SETUP ---
    debug_dir = None
    if args.generate_files:
        debug_dir = input_dir / f"{clean_name}_capsule_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\nDebug mode enabled. Intermediate files will be saved to: {debug_dir}\n")

    def save_debug(key: str, img: sitk.Image):
        if debug_dir:
            out_path = debug_dir / f"{key}.nii.gz"
            logger.info(f"  -> Generating debug file: {out_path}...")
            sitk.WriteImage(img, str(out_path))

    try:
        logger.info(f"\nLoading raw scan '{input_path}'...")
        img = sitk.ReadImage(str(input_path))

        lumen_path = Path(args.lumen_mask).resolve()
        if not lumen_path.exists():
            logger.error(f"The required lumen mask '{lumen_path}' was not found.")
            raise FileNotFoundError(f"Missing lumen mask: {lumen_path}")

        logger.info(f"Loading lumen mask '{lumen_path}'...\n")
        lumen_img = sitk.ReadImage(str(lumen_path))

        if lumen_img.GetSize() != img.GetSize():
            logger.error(
                "Dimension mismatch! The lumen mask does not match the raw scan dimensions.")
            raise ValueError("Dimension mismatch between input and lumen mask.")

        capsule_mask = isolate_capsule(
            img, args.threshold, lumen_img,
            args.min_voxels, args.search_radius, args.max_labels,
            args.patch_radius, save_debug
        )

        logger.info("\nApplying final label and realigning metadata...\n")

        # Cast to UInt16 to prevent overflow if args.label > 255
        final_mask = sitk.Cast(capsule_mask, sitk.sitkUInt16) * args.label
        final_mask.CopyInformation(img)

        # Standard capsule export path
        if args.output:
            capsule_out_path = Path(args.output)
        else:
            capsule_out_path = input_dir / f"{clean_name}_capsule_mask.nii.gz"

        logger.info(f"Saving isolated capsule mask to '{capsule_out_path}'...")
        sitk.WriteImage(final_mask, str(capsule_out_path))

        # Additional combined export path
        if args.combine:
            logger.info("Combining Capsule and Lumen masks into a single volume...")

            # Enforce a safe label for the lumen to ensure it doesn't overwrite the capsule
            safe_lumen_label = args.label + 1 if args.label == 1 else 1
            logger.info(
                f"Assigning Lumen to label {safe_lumen_label} to prevent collision with Capsule (label {args.label}).")

            # Binarize lumen (>0) to strip old labels, multiply by new safe label, cast safely
            lumen_img_16bit = sitk.Cast(lumen_img > 0, sitk.sitkUInt16) * safe_lumen_label

            combined_mask = final_mask + lumen_img_16bit
            combined_mask.CopyInformation(img)

            if args.output:
                if capsule_out_path.name.endswith('.nii.gz'):
                    combined_filename = f"{capsule_out_path.name[:-7]}_combined.nii.gz"
                    combined_out_path = capsule_out_path.with_name(combined_filename)
                else:
                    combined_out_path = capsule_out_path.with_name(
                        f"{capsule_out_path.stem}_combined{capsule_out_path.suffix}")
            else:
                combined_out_path = input_dir / f"{clean_name}_combined_mask.nii.gz"

            logger.info(f"Saving combined mask to '{combined_out_path}'...")
            sitk.WriteImage(combined_mask, str(combined_out_path))

        logger.info("Processing complete!")

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="High-Contrast Prostate Capsule Segmentation."
    )

    parser.add_argument("-i", "--input", required=True, help="Path to raw scan (REQUIRED)")
    parser.add_argument("-m", "--lumen_mask", required=True,
                        help="Path to the lumen mask exclusion zone (REQUIRED)")
    parser.add_argument("-o", "--output", default=None,
                        help="Path to save final mask (Optional)")
    parser.add_argument("-c", "--combine", action="store_true",
                        help="Combine capsule and lumen masks into one file.")
    parser.add_argument("-l", "--label", type=int, default=1,
                        help="Int; Label number to apply to the capsule (default: 1)")
    parser.add_argument("-t", "--threshold", type=float, default=300.0,
                        help="Float; Lower intensity limit for the capsule (default: 300.0)")
    parser.add_argument("-g", "--generate_files", action="store_true",
                        help="Generate intermediate debug files.")
    parser.add_argument("--min_voxels", type=int, default=500,
                        help="Int; Minimum size to prevent locking onto microscopic dust (default: 500)")
    parser.add_argument("--search_radius", type=int, default=15,
                        help="Int; Voxel radius to expand the lumen to reach the capsule (default: 15)")
    parser.add_argument("--max_labels", type=int, default=5,
                        help="Int; Restrict targeting to only the N largest components (default: 5)")
    parser.add_argument("-p", "--patch_radius", type=int, default=25,
                        help="Int; Voxel radius for distance mapping patching (default: 25. Set to 0 to disable).")

    args = parser.parse_args()

    try:
        auto_segment_capsule(args)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()