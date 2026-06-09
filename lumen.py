import argparse
import SimpleITK as sitk
import numpy as np
import sys
import os

def initial_tissue_mask(img, upper_thresh, gen_file):

    print("Generating initial solid tissue mask...")
    solid_mask = img > upper_thresh
    solid_components = sitk.ConnectedComponent(solid_mask)
    largest_solid = sitk.RelabelComponent(solid_components) == 1

    if gen_file:
        print("Generating step1_initial_tissue_body.nii.gz...")
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
        if tissue_ratio > tissue_to_air - 0.045:
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
        print("Generating step2_cropped_raw_image.nii.gz...")
        sitk.WriteImage(cropped_img, "step2_cropped_raw_image.nii.gz")

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
        print("Generating step3_cropped_all_internal_air.nii.gz...")
        sitk.WriteImage(sitk.Cast(internal_air, sitk.sitkUInt8),
                        "step3_cropped_all_internal_air.nii.gz")

    return internal_air

def shave_faces(internal_air, shave_global, shave_top, shave_bottom, gen_file):
    # --- RESOLVE SHAVE PARAMETERS FOR EACH FACE [X, Y, Z] ---
    # Use specific overrides if provided, otherwise fall back to global shave value
    z_shave_bottom = shave_bottom if shave_bottom is not None else shave_global
    z_shave_top = shave_top if shave_top is not None else shave_global

    lower_shave_bounds = [shave_global, shave_global, z_shave_bottom]
    upper_shave_bounds = [shave_global, shave_global, z_shave_top]

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
        print("Generating step4_cropped_isolated_lumen.nii.gz...")
        sitk.WriteImage(sitk.Cast(largest_lumen, sitk.sitkUInt8),
                        "step4_cropped_isolated_lumen.nii.gz")

    return largest_lumen

def auto_segment_lumen(input_path, output_path, upper_thresh, shave_global, shave_top,
                       shave_bottom, gen_file, target_label, tissue_to_air):
    if not os.path.exists(input_path):
        print(f"Error: The input scan '{input_path}' was not found.")
        sys.exit(1)

    try:
        print(f"Loading raw scan '{input_path}'...")
        img = sitk.ReadImage(input_path)
        img_array = sitk.GetArrayFromImage(img)

        # --- STEP 1: INITIAL TISSUE MASK ---
        solid_array = initial_tissue_mask(img, upper_thresh, gen_file)

        # --- STEP 2: THE "GOLDEN SLICE" REFERENCE ---
        min_x, max_x, min_y, max_y, reference_slice = extract_middle(solid_array)

        # --- STEP 3: CALCULATE 4-WAY CROP LINES ---
        crop_top, crop_bottom, crop_left, crop_right = (
            calc_crop(min_x, max_x, min_y, max_y, reference_slice, tissue_to_air))

        # --- STEP 4: EXECUTE THE 3D CROP ---
        cropped_img = perform_crop(img, img_array, crop_top, crop_bottom, crop_left, crop_right,
                                   gen_file)

        # --- STEP 5: SEGMENT AND ASYMMETRICALLY SHAVE ---
        internal_air = segment_air(cropped_img, upper_thresh, gen_file)
        largest_lumen = shave_faces(internal_air, shave_global, shave_top, shave_bottom, gen_file)

        # --- STEP 6: PASTE BACK TO ORIGINAL DIMENSIONS ---
        print("Realigning mask with original dimensions...")
        final_full_mask = sitk.Image(img.GetSize(), sitk.sitkUInt8)
        final_full_mask.CopyInformation(img)

        lumen_mask_8bit = sitk.Cast(largest_lumen, sitk.sitkUInt8) * target_label
        final_full_mask = sitk.Paste(
            destinationImage=final_full_mask,
            sourceImage=lumen_mask_8bit,
            sourceSize=lumen_mask_8bit.GetSize(),
            sourceIndex=[0, 0, 0],
            destinationIndex=[crop_left, crop_top, 0]
        )

        print(f"Saving final result to '{output_path}'...")
        sitk.WriteImage(final_full_mask, output_path)
        print("Processing complete!")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
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
    parser.add_argument("-t", "--tissue_to_air", type=float, default=0.95,
                        help="Float; Tissue-to-air ratio limit (default: 0.95)")

    # Global shave parameter
    parser.add_argument("-s", "--shave", type=int, default=4,
                        help="Int; Global number of voxels to shave off all 6 faces (default: 2)")

    # Specialized face overrides
    parser.add_argument("-st", "--shave_top", type=int, default=10,
                        help="Int; Specific number of voxels to shave from the top face (Z-upper)."
                             " Overrides global -s (default: 10)")
    parser.add_argument("-sb", "--shave_bottom", type=int, default=15,
                        help="Int; Specific number of voxels to shave from the bottom face "
                             "(Z-lower). Overrides global -s (default: 15)")

    # Generate intermediate files (for debugging)
    parser.add_argument("-g", "--generate_files", action="store_true",
                        help="Including this flag generates intermediate files (for debugging)")

    args = parser.parse_args()
    auto_segment_lumen(args.input, args.output, args.upper_thresh, args.shave, args.shave_top,
                       args.shave_bottom, args.generate_files, args.label, args.tissue_to_air)


if __name__ == "__main__":
    main()