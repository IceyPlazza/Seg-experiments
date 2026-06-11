# Seg-Experiments

A collection of Python scripts designed for experimenting with ITK-SNAP, SimpleITK, and adaptive 
BPH (Benign Prostatic Hyperplasia) segmentations.

## Prerequisites

Ensure you have Python installed along with the required libraries. You can install the 
dependencies using pip:

```bash
pip install SimpleITK numpy
```

## Usage

### 1. Cleaning Labels (`clean_label.py`)
Used for simply removing disconnected components and isolating the primary segmented body.

#### Basic Command
```bash
python clean_label.py -i /path/to/raw/scan.nii.gz -o /path/to/output/cleaned_mask.nii.gz -l 2 
```

### 2. Adaptive BPH Lumen Segmentation (`lumen.py`)

An automated 7-step pipeline that segments the internal BPH lumen by calculating tissue-to-air 
boundaries, cropping the volume, isolating internal air, and dynamically shaving artifacts based on 
structural geometry (circularity and extent).

#### Basic Command

```bash
python lumen.py -i /path/to/raw/scan.nii.gz
```

#### Generating Debugging Files (To see intermediate steps)

```bash
python lumen.py -i /path/to/scan.nii.gz -g
```

### `lumen.py` Configuration Flags

You can heavily customize the segmentation behavior using the following optional arguments. 
Use `python lumen.py -h` in the terminal for a quick reference.

- `-l` / `--label`: The integer value assigned to the final output mask (Default: `2`).
- `-u` / `--upper_thresh`: Upper threshold limit to partition tissue from air (Default: `-500.0`).
- `-t` / `--tissue_to_air`: Density ratio required to establish the 4-way crop boundaries
(Default: `0.92`).
- `-s` / `--shave`: The base number of voxels to uniformly shave off all non-dynamic faces 
(Default: `5`).
- `-st` / `--shave_top`: Target circularity score (0.0 to 1.0) to stop dynamically shaving slices 
from the top down (Default: `0.5`).
- `-sb` / `--shave_bottom`: Target extent score (0.0 to 1.0) to stop dynamically shaving slices from
the bottom up (Default: `0.4`).
- `-sa` / `--shave_anterior`: Target extent score (0.0 to 1.0) to stop dynamically shaving 
coronal slices inward from the anterior face (Default: `0.2`).
- `-sl` / `--shave_limit`: The maximum number of slices the script is allowed to dynamically shave 
before capping out (Default: `35`).
- `-g` / `--generate_files`: Include this flag to output intermediate steps 
(`step1_...`, `step4_...`, etc.) to your working directory for debugging.

## Troubleshooting

If your final lumen mask isn't looking quite right, try adjusting the corresponding flags:

| Symptom                                                 | Solution                                                                                                            | Flag to Adjust                                  |
|---------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| Lumen is way too small                                  | The crop boundaries are too strict. Lower the tissue-to-air density requirement.                                    | Try decreasing `-t` (e.g., `0.85` or `0.80`)    |
| Tissue included in segmentation / bits of lumen missing | The contrasts of the scans are too light / too dark. Adjust the upper threshold limit.                              | Try adjusting `-u` (e.g., `-600.0` or `-400.0`) |
| Too much trimmed off the sides                          | The dynamic shaver is digging too deep before hitting its target. Lower the maximum slice limit.                    | Try decreasing `-sl` (e.g., `20` or `15`)       |
| Top artifact not trimming                               | The script thinks the artifact is circular enough. Increase the strictness of the circularity target.               | Try increasing `-st` (e.g., `0.6` or `0.7`)     |
| Bottom artifact not trimming                            | The script thinks the artifact matches the extent target. Increase the strictness of the extent target.             | Try increasing `-sb` (e.g., `0.5` or `0.6`)     |
| Anterior artifact not trimming                          | The script thinks the artifact is semi-circular enough. Increase the strictness of the anterior circularity target. | Try increasing `-sa` (e.g., `0.4` or `0.5`)     |
