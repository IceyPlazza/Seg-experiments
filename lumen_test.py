"""TEMPORARY comparison harness — Otsu-adaptive air/tissue threshold vs. the fixed -450.

Not part of the pipeline; safe to delete. Reuses the real functions from
``autosegct.bph.lumen`` so the only thing that varies between runs is the
``upper_thresh`` value fed into them.

It reports, for each candidate threshold:
  * the threshold value (HU),
  * the largest solid-tissue component (Step 1),
  * the isolated internal-air volume (Step 3),
  * and, with --full, the final shaved lumen volume (Step 4) plus its Dice
    overlap against the current -450 baseline.

With --full, the final lumen mask for every candidate is also written to a
``<case>_lumen_test/`` directory next to the scan (label value = LUMEN_LABEL,
same as the real pipeline) so all three can be loaded into a viewer and
compared visually. Without --full only the fast Step 1/3 counts are printed
(no shave, no files) for a quick threshold peek.

Candidates compared:
  * baseline    : the current hardcoded default (-450, or whatever -u you pass)
  * otsu        : whole-image 2-class Otsu (the naive drop-in)
  * otsu_multi  : lower boundary of a 3-class Otsu (air / tissue / dense) —
                  robust to the dense capsule tail skewing a 2-class split

Usage:
  python lumen_test.py -i path/to/scan.nii.gz            # fast threshold peek
  python lumen_test.py -i path/to/scan.nii.gz --full     # full pipeline + masks + Dice
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from autosegct.utils.io import make_step_saver, save_sitk_image, derive_clean_name
from autosegct.bph.labels import LUMEN_LABEL
from autosegct.bph.lumen import (
    initial_tissue_mask, crop_to_phantom, segment_air,
    calc_top_shave, calc_bottom_shave, calc_anterior_shave, shave_faces,
)

# Shave/crop params fixed at the main_bph.py defaults so threshold is the only variable.
SHAVE_DEFAULTS = dict(
    tissue_to_air=0.92, shave=5, shave_top=0.3, shave_bottom=0.4,
    shave_anterior=0.4, shave_limit=35,
)

# A no-op saver: we only care about voxel counts here, not debug dumps.
_SILENT = make_step_saver(None)


def otsu_threshold(img):
    """Whole-image 2-class Otsu; returns the HU boundary."""
    f = sitk.OtsuThresholdImageFilter()
    f.Execute(img)
    return f.GetThreshold()


def otsu_multi_lower(img, n=2):
    """Lower boundary of an (n+1)-class Otsu split (air / tissue / dense...)."""
    f = sitk.OtsuMultipleThresholdsImageFilter()
    f.SetNumberOfThresholds(n)
    f.Execute(img)
    return list(f.GetThresholds())


def paste_to_full(cropped_bool, full_shape, crop_left, crop_top):
    """Place a cropped boolean lumen array back into the original volume frame."""
    full = np.zeros(full_shape, dtype=bool)
    d, h, w = cropped_bool.shape
    full[0:d, crop_top:crop_top + h, crop_left:crop_left + w] = cropped_bool
    return full


def dice(a, b):
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return (2.0 * inter / denom) if denom else 1.0


def run_one(img, img_array, thresh, full):
    """Run the real lumen steps at a given threshold; return a metrics dict.

    Always runs Steps 1/3 (solid tissue + internal air). When *full*, also runs
    Step 4 (asymmetric shave) and pastes the final lumen back into the original
    frame so it can be written out and Dice-compared.
    """
    args = SimpleNamespace(upper_thresh=float(thresh), **SHAVE_DEFAULTS)

    solid_array = initial_tissue_mask(img, args.upper_thresh, _SILENT)
    solid_voxels = int(solid_array.sum())

    cropped_img, crop_left, crop_top = crop_to_phantom(
        img, img_array, solid_array, args.tissue_to_air, _SILENT)

    internal_air = segment_air(cropped_img, args.upper_thresh, _SILENT)
    air_voxels = int(sitk.GetArrayViewFromImage(internal_air).sum())

    result = dict(thresh=float(thresh), solid=solid_voxels, air=air_voxels)

    if full:
        top = calc_top_shave(internal_air, args.shave_top, args.shave_limit)
        bot = calc_bottom_shave(internal_air, args.shave_bottom, args.shave_limit)
        ant = calc_anterior_shave(internal_air, args.shave_anterior, args.shave_limit, top, bot)
        lumen = shave_faces(internal_air, args.shave, top, bot, ant, _SILENT)
        lumen_arr = sitk.GetArrayFromImage(lumen).astype(bool)
        result["lumen"] = int(lumen_arr.sum())
        result["full_mask"] = paste_to_full(lumen_arr, img_array.shape, crop_left, crop_top)

    return result


def write_mask(full_bool, ref_img, out_path):
    """Write a full-frame boolean lumen mask as a LUMEN_LABEL NIfTI with scan metadata."""
    mask_img = sitk.Cast(sitk.GetImageFromArray(full_bool.astype(np.uint8)), sitk.sitkUInt8)
    mask_img = mask_img * LUMEN_LABEL
    save_sitk_image(mask_img, str(out_path), ref_img)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True, help="Path to raw scan (REQUIRED)")
    ap.add_argument("-u", "--upper_thresh", type=float, default=-450.0,
                    help="Baseline fixed threshold to compare against (default: -450)")
    ap.add_argument("--full", action="store_true",
                    help="Run the full pipeline through the shave, write each candidate's "
                         "lumen mask to <case>_lumen_test/, and report final lumen + Dice")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    clean_name = derive_clean_name(input_path)
    out_dir = input_path.parent / f"{clean_name}_lumen_test"
    if args.full:
        out_dir.mkdir(exist_ok=True)

    print(f"Loading {input_path} ...")
    img = sitk.ReadImage(str(input_path))
    img_array = sitk.GetArrayFromImage(img)

    lo, hi = float(img_array.min()), float(img_array.max())
    pcts = np.percentile(img_array, [1, 5, 50, 95, 99])
    print(f"Intensity range: [{lo:.0f}, {hi:.0f}] HU")
    print("Percentiles [1,5,50,95,99]: " + ", ".join(f"{p:.0f}" for p in pcts))

    t_otsu = otsu_threshold(img)
    multi = otsu_multi_lower(img, n=2)
    print(f"\nWhole-image Otsu (2-class): {t_otsu:.1f} HU")
    print(f"Otsu 3-class boundaries:    {', '.join(f'{m:.1f}' for m in multi)} HU "
          f"(using lower = {multi[0]:.1f})")
    print(f"Current baseline:           {args.upper_thresh:.1f} HU\n")

    candidates = [
        ("baseline", args.upper_thresh),
        ("otsu", t_otsu),
        ("otsu_multi", multi[0]),
    ]

    results = {}
    for name, thr in candidates:
        print(f"\n===== {name}: threshold = {thr:.1f} HU =====")
        r = run_one(img, img_array, thr, args.full)
        if args.full:
            out_path = out_dir / f"{clean_name}_lumen_{name}.nii.gz"
            write_mask(r["full_mask"], img, out_path)
            print(f"  -> wrote {out_path}")
        results[name] = r

    # --- Summary table ---
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = f"{'candidate':<12}{'thresh':>10}{'solid':>14}{'internal_air':>16}"
    if args.full:
        header += f"{'lumen':>14}{'dice_vs_base':>14}"
    print(header)

    base_mask = results["baseline"].get("full_mask") if args.full else None
    for name, _ in candidates:
        r = results[name]
        row = f"{name:<12}{r['thresh']:>10.1f}{r['solid']:>14,}{r['air']:>16,}"
        if args.full:
            d = dice(base_mask, r["full_mask"])
            row += f"{r['lumen']:>14,}{d:>14.4f}"
        print(row)

    if args.full:
        print("\ndice_vs_base = overlap of each candidate's final lumen against the "
              "baseline (-u) lumen.")
        print("1.0000 means the threshold change made no difference to the final mask.")
        print(f"\nMasks written to: {out_dir}")
    else:
        print("\n(Run with --full to shave to completion, write the masks, and get Dice.)")


if __name__ == "__main__":
    sys.exit(main())
