"""TEMPORARY comparison harness — soft-tissue-anchored capsule threshold vs. the fixed 300.

Not part of the pipeline; safe to delete. Reuses the real functions from
``autosegct.bph.hi.capsule`` so the only thing that varies between runs is the
``lower_thresh`` value fed into them. Requires the lumen mask to already exist
(the capsule stage anchors on it), so run the lumen pipeline first.

Whole-image Otsu was already ruled out for the capsule: it is a thin, bright tail with
no mass-balanced histogram valley, so Otsu's boundaries land down in soft tissue and the
"capsule" explodes to tens of millions of voxels. This harness instead tests thresholds
**anchored to the soft-tissue (phantom-body) distribution**, which tracks scan-to-scan
brightness drift while keeping the capsule as the bright tail above it:

  * The phantom body is isolated with the lumen stage's own machinery
    (`resolve_upper_thresh` -> Otsu air/tissue boundary; `initial_tissue_mask` -> largest
    solid component), and HU statistics are drawn from those body voxels.

Candidates compared:
  * baseline : the current hardcoded default (300, or whatever --lower_thresh you pass)
  * st_mad   : median(body) + k_mad * MAD(body)   -- robust to the capsule's own bright tail
  * st_std   : mean(body)   + k_std * std(body)   -- classic, but std is inflated by the tail

Tune --k_mad / --k_std to reproduce your working ~300 (a calibration table is printed).

Usage:
  python capsule_test.py -i path/to/scan.nii.gz                       # fast threshold peek
  python capsule_test.py -i path/to/scan.nii.gz --full                # full isolate + masks + Dice
  python capsule_test.py -i path/to/scan.nii.gz --full --k_mad 6 --k_std 2.5
  python capsule_test.py -i path/to/scan.nii.gz --full --sweep 2 2.5 3  # one st_mad per k, vs baseline
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from autosegct.utils.io import (
    make_step_saver, save_sitk_image, derive_clean_name, mask_path)
from autosegct.bph.labels import CAPSULE_LABEL, LUMEN_LABEL, BPH_LABEL_NAMES
from autosegct.bph.lumen import resolve_upper_thresh, initial_tissue_mask
from autosegct.bph.hi.capsule import (
    create_exclusion_masks, extract_largest_components, isolate_capsule)

# Geometric/morphological params fixed at the main_bph.py capsule defaults so the
# intensity threshold is the only variable across candidates.
CAPSULE_DEFAULTS = dict(min_voxels=500, search_radius=15, max_labels=5, patch_radius=30)

_SILENT = make_step_saver(None)


def body_stats(img, img_array):
    """Isolate the phantom body (largest solid tissue) and return its HU statistics.

    Uses the lumen stage's own air/tissue Otsu split + largest-component extraction so the
    'soft tissue' population is defined identically to how the pipeline already sees it.
    """
    air_thresh = resolve_upper_thresh(img, None)  # Otsu air/tissue boundary (logged)
    body = initial_tissue_mask(img, air_thresh, _SILENT) > 0
    vals = img_array[body]
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    mean = float(vals.mean())
    std = float(vals.std())
    pcts = {p: float(np.percentile(vals, p)) for p in (50, 95, 99, 99.5, 99.9)}
    return dict(air_thresh=air_thresh, n=int(body.sum()),
                median=median, mad=mad, mean=mean, std=std, pcts=pcts)


def dice(a, b):
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return (2.0 * inter / denom) if denom else 1.0


def run_one(img, lumen_img, thresh, full):
    """Run the real capsule steps at a given threshold; return a metrics dict.

    Always runs Steps 1/2 (threshold + largest components). When *full*, runs the
    complete isolate_capsule (centroid lock + bridge + seal) so it can be written
    and Dice-compared.
    """
    result = dict(thresh=float(thresh))

    binary_mask, _lumen_binary, _safe = create_exclusion_masks(
        img, float(thresh), lumen_img, _SILENT)
    result["raw"] = int(sitk.GetArrayViewFromImage(binary_mask).sum())

    _comps, cc_stats = extract_largest_components(
        binary_mask, CAPSULE_DEFAULTS["max_labels"], _SILENT)
    labels = [l for l in cc_stats.GetLabels() if l > 0]
    result["largest"] = max((cc_stats.GetNumberOfPixels(l) for l in labels), default=0)

    if full:
        capsule = isolate_capsule(
            img, float(thresh), lumen_img,
            CAPSULE_DEFAULTS["min_voxels"], CAPSULE_DEFAULTS["search_radius"],
            CAPSULE_DEFAULTS["max_labels"], CAPSULE_DEFAULTS["patch_radius"], _SILENT)
        arr = sitk.GetArrayFromImage(capsule).astype(bool)
        result["capsule"] = int(arr.sum())
        result["mask"] = arr

    return result


def write_mask(mask_bool, ref_img, out_path):
    """Write a boolean capsule mask as a CAPSULE_LABEL NIfTI with scan metadata."""
    mask_img = sitk.Cast(sitk.GetImageFromArray(mask_bool.astype(np.uint8)), sitk.sitkUInt8)
    mask_img = mask_img * CAPSULE_LABEL
    save_sitk_image(mask_img, str(out_path), ref_img)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True, help="Path to raw scan (REQUIRED)")
    ap.add_argument("--lower_thresh", type=float, default=300.0,
                    help="Baseline fixed threshold to compare against (default: 300)")
    ap.add_argument("--k_mad", type=float, default=5.0,
                    help="Multiplier for median + k*MAD candidate (default: 5.0)")
    ap.add_argument("--k_std", type=float, default=3.0,
                    help="Multiplier for mean + k*std candidate (default: 3.0)")
    ap.add_argument("--sweep", type=float, nargs="+", default=None, metavar="K",
                    help="One or more k_mad values to sweep in a single run; each becomes an "
                         "st_mad_k<K> candidate (median + K*MAD) compared against baseline. "
                         "Overrides --k_mad/--k_std. E.g. --sweep 2 2.5 3")
    ap.add_argument("--full", action="store_true",
                    help="Run the full isolate_capsule, write each candidate's mask to "
                         "<case>_capsule_test/, and report final capsule volume + Dice")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    clean_name = derive_clean_name(input_path)
    input_dir = input_path.parent
    out_dir = input_dir / f"{clean_name}_capsule_test"
    if args.full:
        out_dir.mkdir(exist_ok=True)

    print(f"Loading {input_path} ...")
    img = sitk.ReadImage(str(input_path))

    lumen_path = mask_path(input_dir, clean_name, BPH_LABEL_NAMES[LUMEN_LABEL])
    if not lumen_path.exists():
        print(f"ERROR: lumen mask not found at {lumen_path}. Run the lumen pipeline first.")
        return 1
    print(f"Loading lumen mask {lumen_path} ...")
    lumen_img = sitk.ReadImage(str(lumen_path))

    img_array = sitk.GetArrayFromImage(img)
    st = body_stats(img, img_array)
    del img_array

    print(f"\nPhantom-body HU stats ({st['n']:,} voxels, air/tissue cut > {st['air_thresh']:.1f}):")
    print(f"  median={st['median']:.1f}  MAD={st['mad']:.1f}  "
          f"mean={st['mean']:.1f}  std={st['std']:.1f}")
    print("  percentiles " + ", ".join(f"p{p}={v:.0f}" for p, v in st['pcts'].items()))

    # Calibration table: which k reproduces the working ~300?
    print("\nCalibration (threshold at each k):")
    print(f"  {'k':>4}{'median+k*MAD':>16}{'mean+k*std':>14}")
    for k in (2, 3, 4, 5, 6, 7, 8):
        print(f"  {k:>4}{st['median'] + k * st['mad']:>16.1f}{st['mean'] + k * st['std']:>14.1f}")

    if args.sweep:
        print("\nSweeping median + k*MAD:")
        candidates = [("baseline", args.lower_thresh)]
        for k in args.sweep:
            thr = st['median'] + k * st['mad']
            candidates.append((f"st_mad_k{k:g}", thr))
            print(f"  k_mad={k:g} -> {thr:.1f} HU")
        print(f"Baseline: {args.lower_thresh:.1f} HU\n")
    else:
        t_mad = st['median'] + args.k_mad * st['mad']
        t_std = st['mean'] + args.k_std * st['std']
        print(f"\nUsing k_mad={args.k_mad} -> st_mad = {t_mad:.1f} HU")
        print(f"Using k_std={args.k_std} -> st_std = {t_std:.1f} HU")
        print(f"Baseline: {args.lower_thresh:.1f} HU\n")
        candidates = [
            ("baseline", args.lower_thresh),
            ("st_mad", t_mad),
            ("st_std", t_std),
        ]

    results = {}
    for name, thr in candidates:
        print(f"\n===== {name}: threshold = {thr:.1f} HU =====")
        r = run_one(img, lumen_img, thr, args.full)
        if args.full:
            out_path = out_dir / f"{clean_name}_capsule_{name}.nii.gz"
            write_mask(r["mask"], img, out_path)
            print(f"  -> wrote {out_path}")
        results[name] = r

    # --- Summary table ---
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = f"{'candidate':<12}{'thresh':>10}{'raw_thresh':>14}{'largest_cc':>14}"
    if args.full:
        header += f"{'capsule':>12}{'dice_vs_base':>14}"
    print(header)

    base_mask = results["baseline"].get("mask") if args.full else None
    for name, _ in candidates:
        r = results[name]
        row = f"{name:<12}{r['thresh']:>10.1f}{r['raw']:>14,}{r['largest']:>14,}"
        if args.full:
            d = dice(base_mask, r["mask"])
            row += f"{r['capsule']:>12,}{d:>14.4f}"
        print(row)

    if args.full:
        print("\ndice_vs_base = overlap of each candidate's final capsule against the "
              "baseline (--lower_thresh) capsule.")
        print("1.0000 means the threshold change made no difference to the final mask.")
        print(f"\nMasks written to: {out_dir}")
    else:
        print("\n(Run with --full to isolate to completion, write the masks, and get Dice.)")


if __name__ == "__main__":
    sys.exit(main())
