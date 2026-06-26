import SimpleITK as sitk
import argparse
import sys
import math
import logging
import numpy as np
from pathlib import Path
from typing import Callable

from skimage.morphology import convex_hull_image

from autosegct.utils.io import derive_clean_name, make_step_saver, save_sitk_image, save_label_stls

# --- LOGGER SETUP ---
logger = logging.getLogger(__name__)

CAPSULE_LABEL = 1
LUMEN_LABEL = 2
LOBE_LABEL = 3

# Minimum capsule pixels on a 2D slice before we bother hulling it (skips noise specks
# and degenerate <3-point slices that the convex hull cannot handle).
MIN_SLICE_PX = 50
# Opening radius (voxels) that severs the thin necks where the lobe pokes through gaps in
# the capsule shell; the disconnected leak fragments then drop out when we keep the
# largest component.
OPEN_RADIUS = 1
# Closing radius (voxels) applied afterwards to smooth the surface and bridge small
# concavities left by the opening.
SMOOTH_CLOSE_RADIUS = 5

# =====================================================================
# PIPELINE STAGE 1: CORE ALGORITHM METHODS
# =====================================================================

def fill_capsule_interior(
        capsule_bin: sitk.Image,
        save_debug: Callable[[str, sitk.Image], None]
) -> sitk.Image:
    """Estimate the solid region enclosed by the capsule via a 3-view visual hull.

    The capsule shell is watertight around the sides but open at both (obliquely
    oriented) poles, so a 3D hole-fill leaks straight out and never encloses the
    interior. Instead, on every 2D slice along each of the three axes we take the
    convex hull of the capsule cross-section — which closes the ring into a filled
    disk even where the shell is only a partial arc — and intersect the three
    stacks. The intersection is the maximal solid consistent with all three
    silhouettes: a close approximation of the capsule's interior that needs no axis
    estimation, no resampling, and no lumen-based pole sealing.

    Args:
        capsule_bin: Binary capsule shell mask.
        save_debug: Callback to save intermediate files.

    Returns:
        sitk.Image: Binary solid mask of the capsule plus its enclosed interior.
    """
    logger.info("=== STEP 1: FILL CAPSULE INTERIOR (3-VIEW VISUAL HULL) ===")

    arr = sitk.GetArrayFromImage(capsule_bin).astype(bool)  # (z, y, x)
    capsule_voxels = int(arr.sum())
    interior = np.zeros_like(arr)

    if arr.any():
        # Restrict the (per-slice, all-axis) hull work to the capsule's bounding box.
        bounds = []
        for axis in range(3):
            present = np.any(arr, axis=tuple(a for a in range(3) if a != axis))
            idx = np.where(present)[0]
            bounds.append((int(idx[0]), int(idx[-1]) + 1))
        (z0, z1), (y0, y1), (x0, x1) = bounds
        sub = arr[z0:z1, y0:y1, x0:x1]

        logger.info("Hulling capsule cross-sections along all three axes...")
        # Intersect the three axis hulls in place so at most two are held at once.
        hull = _hull_along_axis(sub, 0)
        hull &= _hull_along_axis(sub, 1)
        hull &= _hull_along_axis(sub, 2)
        interior[z0:z1, y0:y1, x0:x1] = hull
        del hull, sub
    del arr

    interior_voxels = int(interior.sum())
    interior_img = sitk.GetImageFromArray(interior.astype(np.uint8))
    del interior
    interior_img.CopyInformation(capsule_bin)
    save_debug("capsule_interior", interior_img)
    logger.info(f"  -> Filled interior: {interior_voxels} voxels "
                f"(capsule shell: {capsule_voxels} voxels).")
    return interior_img


def _hull_along_axis(volume: np.ndarray, axis: int) -> np.ndarray:
    """Per-slice 2D convex hull of *volume* stacked along *axis*.

    Each slice perpendicular to *axis* is replaced by the convex hull of its True
    pixels (slices with fewer than ``MIN_SLICE_PX`` pixels are left empty). Returns a
    boolean array the same shape as *volume*.
    """
    out = np.zeros_like(volume)
    moved_in = np.moveaxis(volume, axis, 0)
    moved_out = np.moveaxis(out, axis, 0)
    for i in range(moved_in.shape[0]):
        if moved_in[i].sum() < MIN_SLICE_PX:
            continue
        moved_out[i] = convex_hull_image(moved_in[i])
    return out


def extract_lobe(
        interior: sitk.Image,
        capsule_bin: sitk.Image,
        lumen_bin: sitk.Image,
        save_debug: Callable[[str, sitk.Image], None]
) -> sitk.Image:
    """Carve the lobe out of the filled capsule interior.

    Subtracts the capsule shell and lumen from the interior solid, then cleans it:
    a morphological opening severs the thin necks where the lobe pokes through gaps in
    the capsule shell, keeping the largest connected component discards those now-
    disconnected leak fragments, a closing smooths the result, and finally the lobe is
    cut flush against the lumen and capsule (which outrank it).

    Args:
        interior: Solid mask of the capsule plus its enclosed interior (from Step 1).
        capsule_bin: Binary capsule shell mask.
        lumen_bin: Binary lumen mask.
        save_debug: Callback to save intermediate files.

    Returns:
        sitk.Image: Binary lobe-tissue mask.
    """
    logger.info("=== STEP 2: CARVE LOBE FROM CAPSULE INTERIOR ===")

    # Capsule and lumen outrank the lobe (focal lesion > lumen > capsule > lobe), so the
    # lobe is excluded from both — once up front and again after the closing. Build that
    # exclusion mask once and reuse it.
    not_barrier = sitk.Not(sitk.Or(capsule_bin, lumen_bin))

    logger.info("Subtracting capsule shell and lumen from the interior...")
    lobe = sitk.And(interior, not_barrier)
    save_debug("lobe_raw", lobe)

    logger.info(f"Opening (radius {OPEN_RADIUS}) to sever thin leak bridges...")
    lobe = sitk.BinaryMorphologicalOpening(lobe, [OPEN_RADIUS] * 3, sitk.sitkBall)

    logger.info("Keeping the largest connected component...")
    lobe = sitk.RelabelComponent(sitk.ConnectedComponent(lobe, False)) == 1
    save_debug("lobe_largest", lobe)

    logger.info(f"Closing (radius {SMOOTH_CLOSE_RADIUS}) to smooth the surface...")
    lobe = sitk.BinaryMorphologicalClosing(lobe, [SMOOTH_CLOSE_RADIUS] * 3, sitk.sitkBall)

    # The smoothing closing may have expanded the lobe a few voxels into the lumen or
    # capsule; cut it flush again so the standalone mask stays non-overlapping.
    logger.info("Cutting lobe flush against lumen and capsule (higher priority)...")
    lobe = sitk.And(lobe, not_barrier)
    del not_barrier

    save_debug("lobe_mask", lobe)
    return lobe


# =====================================================================
# PIPELINE STAGE 2: ORCHESTRATION & EXPORT
# =====================================================================

def isolate_lobe(
        capsule_img: sitk.Image,
        lumen_img: sitk.Image,
        save_debug: Callable[[str, sitk.Image], None]
) -> sitk.Image:
    """Orchestrate the lobe isolation: fill the capsule interior, then carve the lobe."""
    capsule_bin = capsule_img > 0
    lumen_bin = lumen_img > 0

    interior = fill_capsule_interior(capsule_bin, save_debug)
    lobe = extract_lobe(interior, capsule_bin, lumen_bin, save_debug)
    del interior, capsule_bin, lumen_bin

    return lobe


def export_results(
        lobe_mask: sitk.Image,
        capsule_img: sitk.Image,
        lumen_img: sitk.Image,
        img: sitk.Image,
        args: argparse.Namespace,
        input_dir: Path,
        clean_name: str
) -> None:
    """Write the standalone lobe mask, the combined mask, and optional STL meshes.

    The combined mask follows the project precedence lumen (2) > capsule (1) > lobe (3);
    overlaps are negligible by construction since the lobe excludes both inputs.
    """
    logger.info("Applying final label and realigning metadata...")
    labeled_mask = sitk.Cast(lobe_mask, sitk.sitkUInt16) * LOBE_LABEL
    labeled_mask.CopyInformation(img)

    # 1. Standalone lobe mask
    lobe_out_path = (
        Path(args.output) if getattr(args, 'output', None)
        else input_dir / f"{clean_name}_lobe_mask.nii.gz"
    )
    logger.info(f"Saving isolated lobe mask to '{lobe_out_path}'...")
    save_sitk_image(labeled_mask, str(lobe_out_path), img)

    # 2. Combined mask. Priority (low -> high, later writes win): lobe < capsule < lumen.
    #    The full project order is focal lesion > lumen > capsule > lobe; a focal-lesion
    #    label (4) would be written last, after lumen, once that stage exists.
    logger.info(
        f"Combining masks: Capsule=label {CAPSULE_LABEL}, Lumen=label {LUMEN_LABEL}, "
        f"Lobe=label {LOBE_LABEL}...")
    combined_arr = np.zeros(sitk.GetArrayViewFromImage(img).shape, dtype=np.uint16)
    combined_arr[sitk.GetArrayViewFromImage(lobe_mask) > 0] = LOBE_LABEL
    combined_arr[sitk.GetArrayViewFromImage(capsule_img) > 0] = CAPSULE_LABEL
    combined_arr[sitk.GetArrayViewFromImage(lumen_img) > 0] = LUMEN_LABEL
    combined_mask = sitk.GetImageFromArray(combined_arr)
    del combined_arr
    combined_mask.CopyInformation(img)

    combined_out_path = input_dir / f"{clean_name}_combined_masks_test.nii.gz"
    logger.info(f"Saving combined mask to '{combined_out_path}'...")
    save_sitk_image(combined_mask, str(combined_out_path), img)

    # 3. STL meshes (if requested)
    if args.stl:
        stl_dir = str(input_dir / f"{clean_name}_stl")
        logger.info(f"Generating lobe STL mesh in: {stl_dir}...")
        save_label_stls(labeled_mask, stl_dir, label_names={LOBE_LABEL: "lobe"})


def auto_segment_lobe(args: argparse.Namespace) -> None:
    """Validate inputs, run the lobe isolation pipeline, and export the results.

    Reads the capsule mask (label 1) and lumen mask (label 2) produced by the upstream
    pipelines, both of which must already exist alongside the input scan.
    """
    logger.info("INITIATING LOBE SEGMENTATION")
    input_path = Path(args.input).resolve()
    input_dir = input_path.parent
    clean_name = derive_clean_name(input_path)

    debug_dir = None
    if args.generate_files:
        debug_dir = str(input_dir / f"{clean_name}_debug" / f"{clean_name}_lobe")
        logger.info(f"Debug mode enabled. Intermediate files will be saved to: {debug_dir}")

    try:
        if not input_path.exists():
            raise FileNotFoundError(f"The input scan '{input_path}' was not found.")

        logger.info(f"Loading raw scan '{input_path}'...")
        img = sitk.ReadImage(str(input_path))

        save_debug = make_step_saver(debug_dir)

        lumen_path = input_dir / f"{clean_name}_lumen_mask.nii.gz"
        capsule_path = input_dir / f"{clean_name}_capsule_mask.nii.gz"
        for required, label in ((lumen_path, "lumen"), (capsule_path, "capsule")):
            if not required.exists():
                raise FileNotFoundError(
                    f"Expected {label} mask '{required}' not found. "
                    f"Run the {label} pipeline first."
                )

        logger.info(f"Loading lumen mask '{lumen_path}'...")
        lumen_img = sitk.ReadImage(str(lumen_path))
        logger.info(f"Loading capsule mask '{capsule_path}'...")
        capsule_img = sitk.ReadImage(str(capsule_path))

        _validate_geometry(img, lumen_img, "lumen")
        _validate_geometry(img, capsule_img, "capsule")

        lobe_mask = isolate_lobe(capsule_img, lumen_img, save_debug)

        if sitk.GetArrayViewFromImage(lobe_mask).sum() == 0:
            logger.warning(
                "Isolated lobe is empty — confirm the capsule mask is present and forms "
                "rings around the lumen.")

        export_results(lobe_mask, capsule_img, lumen_img, img, args, input_dir, clean_name)

        logger.info("Processing complete!")

    except Exception as e:
        logger.exception(f"An error occurred during processing: {e}")
        sys.exit(1)


def _validate_geometry(img: sitk.Image, mask: sitk.Image, name: str) -> None:
    """Confirm *mask* shares the scan's size, spacing, and origin."""
    if mask.GetSize() != img.GetSize():
        raise ValueError(f"Dimension mismatch between input scan and {name} mask.")
    if not all(math.isclose(a, b, rel_tol=1e-3)
               for a, b in zip(img.GetSpacing(), mask.GetSpacing())):
        raise ValueError(f"Spacing mismatch between input scan and {name} mask.")
    if not all(math.isclose(a, b, abs_tol=0.5)
               for a, b in zip(img.GetOrigin(), mask.GetOrigin())):
        raise ValueError(f"Origin mismatch between input scan and {name} mask.")


# =====================================================================
# STANDALONE CLI (for isolated testing; main entry point is main_bph.py)
# =====================================================================

class _ColorFormatter(logging.Formatter):
    """Colorize log output for the standalone CLI (mirrors main_bph.py)."""
    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"
    _GREEN  = "\033[92m"   # INITIATING lines
    _CYAN   = "\033[96m"   # Step headers
    _YELLOW = "\033[93m"   # WARNING
    _RED    = "\033[91m"   # ERROR

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if not sys.stderr.isatty():
            return msg
        if record.levelno >= logging.ERROR:
            return self._RED + msg + self._RESET
        if record.levelno == logging.WARNING:
            return self._YELLOW + msg + self._RESET
        text = record.getMessage()
        if text.startswith("INITIATING"):
            return self._BOLD + self._GREEN + msg + self._RESET
        if text.startswith("==="):
            return self._BOLD + self._CYAN + msg + self._RESET
        if text.startswith("Processing complete"):
            return self._BOLD + self._GREEN + msg + self._RESET
        return msg


def main():
    parser = argparse.ArgumentParser(
        description="Standalone BPH lobe segmentation (fills lobe tissue from the "
                    "capsule and lumen masks). Expects <case>_lumen_mask.nii.gz and "
                    "<case>_capsule_mask.nii.gz next to the input scan.")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to raw scan (REQUIRED). Used only for spatial metadata "
                             "and to locate the sibling lumen/capsule masks.")
    parser.add_argument("-o", "--output", default=None,
                        help="Optional explicit path for the standalone lobe mask "
                             "(default: <case>_lobe_mask.nii.gz beside the input).")
    parser.add_argument("-g", "--generate_files", action="store_true",
                        help="Save numbered intermediate debug files to "
                             "<case>_debug/<case>_lobe/.")
    parser.add_argument("--stl", action="store_true",
                        help="Generate a 3D printable STL mesh of the lobe.")

    args = parser.parse_args()
    _handler = logging.StreamHandler()
    _handler.setFormatter(_ColorFormatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[_handler])

    auto_segment_lobe(args)


if __name__ == "__main__":
    main()
