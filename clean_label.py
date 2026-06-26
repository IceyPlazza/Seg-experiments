import SimpleITK as sitk
import argparse
import sys
import os


def clean_specific_label(input_path, output_path, target_label):
    # Check if the input file actually exists before processing
    if not os.path.exists(input_path):
        print(f"Error: The input file '{input_path}' was not found.")
        sys.exit(1)

    try:
        # Load the multi-label segmentation image
        print(f"Loading '{input_path}'...")
        img = sitk.ReadImage(input_path)

        # SPLIT: isolate target label and preserve the rest

        # Create a boolean mask of ONLY the target label
        is_target = img == target_label

        # Create a mask of everything that is not target label
        is_not_target = sitk.Cast(img != target_label, img.GetPixelID())
        other_labels = img * is_not_target

        # CLEAN
        print(f"Removing disconnected components for label {target_label}...")

        # Snap thin bridges by applying Morphological Opening (radius of 2 voxels)
        # This will erode the image by 2 voxel, then dilate it by 2 voxels.
        opened_target = sitk.BinaryMorphologicalOpening(is_target, (1, 1, 1))

        # Run the component analysis on the opened target, with strict face-connectivity
        connected_components = sitk.ConnectedComponent(opened_target, False)

        # Sort and keep the largest component
        sorted_components = sitk.RelabelComponent(connected_components)
        largest_component = sorted_components == 1

        # Clean and reconnect in case too much details trimmed
        largest_component = sitk.BinaryMorphologicalClosing(largest_component, (5, 5, 5))

        # Cast back to original type and label
        cleaned_target = sitk.Cast(largest_component, img.GetPixelID()) * target_label

        # RECOMBINE
        final_img = other_labels + cleaned_target

        # SAVE
        print(f"Saving to '{output_path}'...")
        sitk.WriteImage(final_img, output_path)
        print("Done!")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        sys.exit(1)


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Clean disconnected components (dust) from a specific label in a multi-label segmentation mask.",
        epilog="Example usage: python clean_label.py -i input.nii.gz -l 2"
    )

    # Define the flags
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input NIfTI segmentation file."
    )

    parser.add_argument(
        "-o", "--output",
        required=False,
        help="Path to save the cleaned NIfTI file (Optional). Defaults to <input>_label_<num>_cleaned in the same directory."
    )

    parser.add_argument(
        "-l", "--label",
        type=int,
        default=1,
        help="The integer value of the label to clean (default: 1)."
    )

    # Parse the arguments from the terminal
    args = parser.parse_args()

    # If output path is not specified, generate it automatically
    if not args.output:
        dir_name = os.path.dirname(args.input)
        base_name = os.path.basename(args.input)

        # Handle the common double extension for NIfTI files
        if base_name.lower().endswith('.nii.gz'):
            name_part = base_name[:-7]
            ext_part = '.nii.gz'
        else:
            name_part, ext_part = os.path.splitext(base_name)

        new_base_name = f"{name_part}_label_{args.label}_cleaned{ext_part}"
        args.output = os.path.join(dir_name, new_base_name)

    # Run the processing function with the provided arguments
    clean_specific_label(args.input, args.output, args.label)


# Standard Python boilerplate to call the main function
if __name__ == "__main__":
    main()