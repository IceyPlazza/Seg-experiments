import argparse
import SimpleITK as sitk
import numpy as np
import sys
import os


def auto_segment_lumen(input_path, output_path, upper_thresh, shave_global, shave_top_val,
                       shave_bottom_val, target_label=1):
    if not os.path.exists(input_path):
        print(f"Error: The input scan '{input_path}' was not found.")
        sys.exit(1)

    try:
        print(f"Loading raw scan '{input_path}'...")
        img = sitk.ReadImage(input_path)
        img_array = sitk.GetArrayFromImage(img)

        # --- STEP 1: INITIAL TISSUE MASK ---
        print("Generating initial solid tissue mask...")
        solid_mask = img > upper_thresh
        solid_components = sitk.ConnectedComponent(solid_mask)
        largest_solid = sitk.RelabelComponent(solid_components) == 1

        sitk.WriteImage(sitk.Cast(largest_solid, sitk.sitkUInt8), "step1_initial_solid_body.nii.gz")
        solid_array = sitk.GetArrayFromImage(largest_solid)

        # --- STEP 2: THE "GOLDEN SLICE" REFERENCE ---
        print("Extracting the middle Z-slice as the clean reference blueprint...")
        mid_z = solid_array.shape[0] // 2
        reference_slice = solid_array[mid_z, :, :]

        valid_y, valid_x = np.where(reference_slice)
        if len(valid_y) == 0 or len(valid_x) == 0:
            print("Error: No solid phantom tissue detected on the middle slice.")
            sys.exit(1)

        min_y, max_y = np.min(valid_y), np.max(valid_y)
        min_x, max_x = np.min(valid_x), np.max(valid_x)

        # --- STEP 3: CALCULATE 4-WAY CROP LINES ---
        print("Calculating boundaries using the 4-sided 95% tissue rule...")
        crop_bottom = int(max_y)
        for y in range(max_y, min_y - 1, -1):
            row_segment = reference_slice[y, min_x:max_x + 1]
            tissue_ratio = np.sum(row_segment) / len(row_segment)
            if tissue_ratio > 0.95:
                crop_bottom = int(y)
                break

        crop_top = int(min_y)
        for y in range(min_y, crop_bottom + 1):
            row_segment = reference_slice[y, min_x:max_x + 1]
            tissue_ratio = np.sum(row_segment) / len(row_segment)
            if tissue_ratio > 0.905:
                crop_top = int(y)
                break

        crop_left = int(min_x)
        for x in range(min_x, max_x + 1):
            column_segment = reference_slice[crop_top:crop_bottom + 1, x]
            tissue_ratio = np.sum(column_segment) / len(column_segment)
            if tissue_ratio > 0.95:
                crop_left = int(x)
                break

        crop_right = int(max_x)
        for x in range(max_x, min_x - 1, -1):
            column_segment = reference_slice[crop_top:crop_bottom + 1, x]
            tissue_ratio = np.sum(column_segment) / len(column_segment)
            if tissue_ratio > 0.95:
                crop_right = int(x)
                break

        # --- STEP 4: EXECUTE THE 3D CROP ---
        print(f"Executing 3D Crop: X({crop_left} to {crop_right}), Y({crop_top} to {crop_bottom})")
        cropped_array = img_array[:, crop_top:crop_bottom + 1, crop_left:crop_right + 1]

        cropped_img = sitk.GetImageFromArray(cropped_array)
        cropped_img.SetSpacing(img.GetSpacing())
        cropped_img.SetDirection(img.GetDirection())
        cropped_img.SetOrigin(img.TransformIndexToPhysicalPoint([crop_left, crop_top, 0]))

        sitk.WriteImage(cropped_img, "step2_cropped_raw_image.nii.gz")

        # --- STEP 5: SEGMENT AND ASYMMETRICALLY SHAVE ---
        print("Segmenting trapped air within the cropped boundaries...")
        cropped_tissue_mask = cropped_img > upper_thresh
        padded_tissue = sitk.ConstantPad(cropped_tissue_mask, [1, 1, 1], [1, 1, 1], 1)

        filled_tissue = sitk.BinaryFillhole(padded_tissue)
        sealed_shell = sitk.Crop(filled_tissue, [1, 1, 1], [1, 1, 1])

        air_mask = cropped_img <= upper_thresh
        internal_air = air_mask * sealed_shell
        sitk.WriteImage(sitk.Cast(internal_air, sitk.sitkUInt8),
                        "step3_all_internal_air_cropped.nii.gz")

        # --- RESOLVE SHAVE PARAMETERS FOR EACH FACE [X, Y, Z] ---
        # Use specific overrides if provided, otherwise fall back to global shave value
        z_shave_bottom = shave_bottom_val if shave_bottom_val is not None else shave_global
        z_shave_top = shave_top_val if shave_top_val is not None else shave_global

        lower_shave_bounds = [shave_global, shave_global, z_shave_bottom]
        upper_shave_bounds = [shave_global, shave_global, z_shave_top]

        print(
            f"Applying Boundary Erosion -> Lower Bounds [X,Y,Z]: {lower_shave_bounds}, Upper Bounds [X,Y,Z]: {upper_shave_bounds}")
        shaved_air = sitk.Crop(internal_air, lower_shave_bounds, upper_shave_bounds)

        # Pad back precisely what was shaved using the asymmetric maps to maintain matrix dimensions
        disconnected_air = sitk.ConstantPad(shaved_air, lower_shave_bounds, upper_shave_bounds, 0)

        print("Filtering disconnected components...")
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.SetFullyConnected(False)
        air_components = cc_filter.Execute(disconnected_air)
        largest_lumen = sitk.RelabelComponent(air_components) == 1

        sitk.WriteImage(sitk.Cast(largest_lumen, sitk.sitkUInt8),
                        "step4_isolated_lumen_cropped.nii.gz")

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
        description="Adaptive Guillotine Segmentation with Custom Asymmetric Face Shaving.")
    parser.add_argument("-i", "--input", required=True, help="Path to raw scan")
    parser.add_argument("-o", "--output", required=True, help="Path to save final mask")
    parser.add_argument("-u", "--upper_thresh", type=float, default=-500.0,
                        help="Upper intensity limit")

    # Base/Global shave parameter
    parser.add_argument("-s", "--shave", type=int, default=2,
                        help="Global number of boundary voxels to shave off all 6 faces (default: 2)")

    # Specialized face overrides
    parser.add_argument("-st", "--shave_top", type=int, default=None,
                        help="Specific number of voxels to shave from the top face (Z-upper). Overrides global -s.")
    parser.add_argument("-sb", "--shave_bottom", type=int, default=None,
                        help="Specific number of voxels to shave from the bottom face (Z-lower). Overrides global -s.")

    args = parser.parse_args()
    auto_segment_lumen(args.input, args.output, args.upper_thresh, args.shave, args.shave_top,
                       args.shave_bottom)


if __name__ == "__main__":
    main()