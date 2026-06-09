# Seg-Experiments

Collections of Python scripts to experiment with ITK-Snap, SimpleITK, and BPH Segmentations.

## Usage

For simply removing disconnected components:

```bash
python clean_label.py -i /path/to/raw/scans/ -o /path/to/output/scans/ -l label_num_to_clean 
```

For BPH Lumen Segmentation:

```bash
python lumen.py -i /path/to/raw/scans/ -o /path/to/output/scans/
```

Use the following to see more info and optional parameters for each script:

```bash
python clean_label.py -h
```
```bash
python lumen.py -h
```