import SimpleITK as sitk
import argparse
import sys
import math
import logging
from pathlib import Path
from typing import Callable, Tuple, List

# --- LOGGER SETUP ---
logger = logging.getLogger(__name__)

# =====================================================================
# PIPELINE STAGE 1: CORE ALGORITHM METHODS
# =====================================================================

def create_exclusion_masks(
        img: sitk.Image,
        lower_thresh: float,
        lumen_img: sitk.Image,
        save_debug: Callable[[str, sitk.Image], None]
) -> Tuple[sitk.Image, sitk.Image, sitk.Image]:
    """
    Applies an initial intensity threshold and carves out the lumen exclusion zone.

    Args:
        img: The raw input medical scan.
        lower_thresh: The minimum intensity limit for identifying the capsule.
        lumen_img: The binary or labeled mask representing the lumen.
        save_debug: Callback function to save intermediate files.

    Returns:
        Tuple containing:
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

    # Exclude lumen territory mathematically
    binary_mask = binary_mask * safe_zone

    return binary_mask, lumen_binary, safe_zone


def extract_largest_components(
        binary_mask: sitk.Image,
        max_labels: int,
        save_debug: Callable[[str, sitk.Image], None]
) -> Tuple[sitk.Image, sitk.LabelShapeStatisticsImageFilter]:
    """
    Isolates the largest connected structures in the thresholded mask, discarding
    microscopic background noise.

    Args:
        binary_mask: The excluded binary mask generated from Step 1.
        max_labels: The maximum number of top largest components to retain.
        save_debug: Callback function to save intermediate files.

    Returns:
        Tuple containing:
            - final_components (sitk.Image): Labeled image of the top N components.
            - cc_stats (sitk.LabelShapeStatisticsImageFilter): Statistics object containing size/centroid data.
    """
    logger.info("=== STEP 2: Component Extraction & Dust Filter ===")

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)
    raw_components = cc_filter.Execute(binary_mask)

    # Sort components by physical size
    final_components_raw = sitk.RelabelComponent(raw_components)
    del raw_components  # [RAM CLEAR]

    # Threshold out everything except the Top N largest objects
    final_components = sitk.Threshold(
        final_components_raw, lower=1, upper=max_labels, outsideValue=0
    )
    del final_components_raw  # [RAM CLEAR]

    save_debug("2_filtered_components", final_components)

    # Compute spatial statistics on the surviving structures
    cc_stats = sitk.LabelShapeStatisticsImageFilter()
    cc_stats.Execute(final_components)

    return final_components, cc_stats


def lock_onto_target_centroid(
        img: sitk.Image,
        final_components: sitk.Image,
        cc_stats: sitk.LabelShapeStatisticsImageFilter,
        lumen_binary: sitk.Image,
        safe_zone: sitk.Image,
        min_voxels: int,
        search_radius: int,
        max_labels: int,
        save_debug: Callable[[str, sitk.Image], None]
) -> sitk.Image:
    """
    Identifies the true capsule fragment by expanding a search ring from the lumen
    and measuring mathematical proximity to the lumen's center of gravity.

    Args:
        img: The raw input medical scan (used for physical coordinate conversions).
        final_components: Labeled image of the top connected components.
        cc_stats: Statistics filter containing data for the final_components.
        lumen_binary: The binarized lumen mask.
        safe_zone: The non-lumen territory mask.
        min_voxels: Minimum required size (in voxels) for a valid component.
        search_radius: Voxel radius to dilate the lumen into a search net.
        max_labels: The number of top labels currently being evaluated.
        save_debug: Callback function to save intermediate files.

    Returns:
        sitk.Image: A binary mask isolating the locked target capsule fragment.
    """
    logger.info("=== STEP 3: The Centroid Lock (Intelligent Targeting) ===")

    # 1. Cast the search net
    logger.info(f"Using a {search_radius}-voxel search net...")
    dilated_lumen = sitk.BinaryDilate(lumen_binary, (search_radius, search_radius, search_radius))
    search_ring = dilated_lumen * safe_zone
    del dilated_lumen  # [RAM CLEAR]
    save_debug("3_search_ring", search_ring)

    # 2. Identify physical overlapping contacts
    overlap_img = sitk.Mask(final_components, search_ring)
    del search_ring  # [RAM CLEAR]
    save_debug("4_overlap_contacts", sitk.Cast(overlap_img > 0, sitk.sitkUInt8))

    overlap_stats = sitk.LabelShapeStatisticsImageFilter()
    overlap_stats.Execute(overlap_img)
    del overlap_img  # [RAM CLEAR]

    # 3. Calculate Target Center
    lumen_stats = sitk.LabelShapeStatisticsImageFilter()
    lumen_stats.Execute(lumen_binary)
    lumen_labels = lumen_stats.GetLabels()

    if lumen_labels:
        target_center = lumen_stats.GetCentroid(lumen_labels[0])
    else:
        logger.warning("Provided lumen mask is empty! Falling back to absolute image center.")
        size = img.GetSize()
        target_center = img.TransformIndexToPhysicalPoint(
            [size[0] // 2, size[1] // 2, size[2] // 2])

    # 4. Evaluate candidates against the target center
    best_label = 0
    min_dist = float('inf')
    top_labels = [l for l in cc_stats.GetLabels() if 0 < l <= max_labels]

    for label in top_labels:
        if cc_stats.GetNumberOfPixels(label) < min_voxels:
            continue
        if not overlap_stats.HasLabel(label):
            continue

        centroid = cc_stats.GetCentroid(label)
        dist = math.dist(centroid, target_center)

        if dist < min_dist:
            min_dist = dist
            best_label = label

    # 5. Lock and isolate
    if best_label == 0:
        logger.warning(
            "No valid capsule found touching the lumen! Falling back to largest component.")
        target_capsule = final_components == 1
    else:
        logger.info(f"  -> Locked onto capsule (Label {best_label}) via Centroid Proximity.")
        target_capsule = final_components == best_label

    return target_capsule


def bridge_planar_gaps(target_capsule: sitk.Image, patch_radius: int,
                       save_debug: Callable) -> sitk.Image:
    """
    Constructs a massive 2D geometric bridge across sheer anatomical dropouts without
    creating artificial caps over the vertical poles.

    Args:
        target_capsule: The isolated binary mask of the target tissue.
        patch_radius: Size of the closing kernel to span anatomical gaps.
        save_debug: Callback function to save intermediate files.

    Returns:
        sitk.Image: The geometrically bridged tissue mask.
    """
    if patch_radius <= 0:
        return target_capsule

    logger.info("=== STEP 4: Native Planar Watertight Patching ===")
    save_debug("4a_base_target_capsule", sitk.Cast(target_capsule, sitk.sitkUInt8))

    logger.info("Solidifying core capsule...")
    solid_native = sitk.BinaryMorphologicalClosing(target_capsule, (2, 2, 2))

    logger.info(f"Applying massive 2D planar bridge (Radius {patch_radius}) to seal gaps...")
    close_filter = sitk.BinaryMorphologicalClosingImageFilter()
    # Z-radius of 0 physically prevents the patch from building caps over the apex/base
    close_filter.SetKernelRadius([patch_radius, patch_radius, 0])
    close_filter.SetKernelType(sitk.sitkBall)
    bridged_native = close_filter.Execute(solid_native)

    # Melt stacked 2D discs into a continuous watertight 3D log
    bridged_native = sitk.BinaryMorphologicalClosing(bridged_native, (2, 2, 2))
    save_debug("4b_bridged_log", sitk.Cast(bridged_native, sitk.sitkUInt8))

    logger.info("Extracting overlapping shell to guarantee 3D watertightness...")
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelRadius([3, 3, 0])
    erode_filter.SetKernelType(sitk.sitkCross)
    inner_core = erode_filter.Execute(bridged_native)

    shell_native = sitk.And(bridged_native, sitk.Not(inner_core))
    synthetic_bridge = sitk.And(shell_native, sitk.Not(solid_native))

    logger.info("Fusing synthetic shell with original capsule...")
    bridged_capsule = sitk.Or(solid_native, synthetic_bridge)

    return bridged_capsule


def seal_micro_punctures(bridged_capsule: sitk.Image, save_debug: Callable) -> sitk.Image:
    """
    Applies a spherical micro-closing to permanently seal remaining structural
    tunnels, followed by a 1-voxel global dilation to guarantee absolute continuity.

    Args:
        bridged_capsule: The tissue mask after the planar gap bridging phase.
        save_debug: Callback function to save intermediate files.

    Returns:
        sitk.Image: The finalized, watertight capsule mask.
    """
    logger.info("=== STEP 5: Final Tuning & Micro-Sealing ===")

    logger.info("Applying spherical micro-closing (Radius 5) to seal tunnels...")
    close_filter = sitk.BinaryMorphologicalClosingImageFilter()
    close_filter.SetKernelRadius([5, 5, 5])
    close_filter.SetKernelType(sitk.sitkBall)
    final_capsule = close_filter.Execute(bridged_capsule)

    logger.info("Applying 1-voxel global thickness dilation to ensure structural continuity...")
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius([1, 1, 1])
    dilate_filter.SetKernelType(sitk.sitkBall)
    thickened_capsule = dilate_filter.Execute(final_capsule)

    save_debug("5_final_thickened_shell", sitk.Cast(thickened_capsule, sitk.sitkUInt8))

    return thickened_capsule


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
        save_debug: Callable[[str, sitk.Image], None]
) -> sitk.Image:
    """
    Orchestrates the ordered algorithmic steps to isolate and seal the prostate capsule.
    """
    binary_mask, lumen_binary, safe_zone = create_exclusion_masks(
        img, lower_thresh, lumen_img, save_debug)

    final_components, cc_stats = extract_largest_components(
        binary_mask, max_labels, save_debug)
    del binary_mask  # [RAM CLEAR]

    target_capsule = lock_onto_target_centroid(
        img, final_components, cc_stats, lumen_binary, safe_zone,
        min_voxels, search_radius, max_labels, save_debug
    )
    del final_components, cc_stats, safe_zone  # [RAM CLEAR]

    bridged_capsule = bridge_planar_gaps(
        target_capsule, patch_radius, save_debug)
    del target_capsule  # [RAM CLEAR]

    final_watertight_capsule = seal_micro_punctures(
        bridged_capsule, save_debug)
    del bridged_capsule, lumen_binary  # [RAM CLEAR]

    return final_watertight_capsule


def export_results(
        final_mask: sitk.Image,
        lumen_img: sitk.Image,
        img: sitk.Image,
        args: argparse.Namespace,
        input_dir: Path,
        clean_name: str
) -> None:
    """
    Handles formatting metadata, labeling, and writing outputs to the local filesystem.

    Args:
        final_mask: The finalized binary mask to be exported.
        lumen_img: The original lumen mask (used for combined mapping).
        img: The raw input medical scan (used to inherit physical spacing/metadata).
        args: The parsed command-line arguments.
        input_dir: Pathlib object representing the save directory.
        clean_name: Formatted string of the scan name without suffixes.
    """
    logger.info("\nApplying final label and realigning metadata...\n")

    # Cast to UInt16 to prevent overflow if args.label > 255
    labeled_mask = sitk.Cast(final_mask, sitk.sitkUInt16) * args.label
    labeled_mask.CopyInformation(img)

    # 1. Export standard capsule mask
    capsule_out_path = Path(
        args.output) if args.output else input_dir / f"{clean_name}_capsule_mask.nii.gz"

    logger.info(f"Saving isolated capsule mask to '{capsule_out_path}'...")
    sitk.WriteImage(labeled_mask, str(capsule_out_path))

    # 2. Export Combined Mask (if requested)
    if args.combine:
        logger.info("Combining Capsule and Lumen masks into a single volume...")

        # Enforce a safe label for the lumen to ensure it doesn't collide with the capsule
        safe_lumen_label = args.label + 1 if args.label == 1 else 1
        logger.info(
            f"Assigning Lumen to label {safe_lumen_label} to prevent collision with Capsule (label {args.label}).")

        # Re-enforce safe zone on the final mask in case mathematical bloating bled into the lumen
        lumen_binary_16bit = sitk.Cast(lumen_img > 0, sitk.sitkUInt16)
        safe_zone_16bit = sitk.Cast(lumen_img == 0, sitk.sitkUInt16)

        safe_capsule_mask = labeled_mask * safe_zone_16bit
        lumen_img_16bit = lumen_binary_16bit * safe_lumen_label

        # Combine safely and realign metadata
        combined_mask = safe_capsule_mask + lumen_img_16bit
        combined_mask.CopyInformation(img)

        # Handle filename logic
        if args.output:
            combined_filename = f"{capsule_out_path.name.split('.')[0]}_combined.nii.gz"
            combined_out_path = capsule_out_path.with_name(combined_filename)
        else:
            combined_out_path = input_dir / f"{clean_name}_combined_mask.nii.gz"

        logger.info(f"Saving combined mask to '{combined_out_path}'...")
        sitk.WriteImage(combined_mask, str(combined_out_path))


def auto_segment_capsule(args: argparse.Namespace) -> None:
    """
    Validates inputs, initializes the debug environment, runs the core isolation
    pipeline, and delegates the export operations.
    """
    input_path = Path(args.input).resolve()

    if not input_path.exists():
        logger.error(f"The input scan '{input_path}' was not found.")
        raise FileNotFoundError(f"Missing input scan: {input_path}")

    input_dir = input_path.parent
    clean_name = input_path.name

    # Cleanly strip standard medical imaging suffixes (.nii, .nii.gz)
    while Path(clean_name).suffixes:
        clean_name = Path(clean_name).stem

    if clean_name.endswith('_0000'):
        clean_name = clean_name[:-5]

    # --- DEBUG FOLDER SETUP ---
    debug_dir = None
    if args.generate_files:
        debug_dir = input_dir / f"{clean_name}_capsule_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\nDebug mode enabled. Intermediate files saved to: {debug_dir}\n")

    def save_debug(key: str, img_debug: sitk.Image) -> None:
        if debug_dir:
            out_path = debug_dir / f"{key}.nii.gz"
            logger.info(f"  -> Generating debug file: {out_path}...")
            sitk.WriteImage(img_debug, str(out_path))

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
            logger.error("Dimension mismatch! The lumen mask does not match raw scan dimensions.")
            raise ValueError("Dimension mismatch between input and lumen mask.")

        # Run Segmentation Pipeline
        capsule_mask = isolate_capsule(
            img, args.threshold, lumen_img,
            args.min_voxels, args.search_radius, args.max_labels,
            args.patch_radius, save_debug
        )

        # Run Export Pipeline
        export_results(capsule_mask, lumen_img, img, args, input_dir, clean_name)

        logger.info("Processing complete!")

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        raise


def main():
    # Setup global logger settings
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

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
    parser.add_argument("-p", "--patch_radius", type=int, default=30,
                        help="Int; Voxel radius for planar patching (default: 30. Set to 0 to disable).")

    args = parser.parse_args()

    try:
        auto_segment_capsule(args)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()