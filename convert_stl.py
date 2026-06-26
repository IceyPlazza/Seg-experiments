import SimpleITK as sitk
import trimesh
import numpy as np
import time


def convert_stl_to_nifti(stl_path, reference_scan_path, output_path):
    print("--> Loading reference scan and STL shell...")
    ref_image = sitk.ReadImage(reference_scan_path)
    mesh = trimesh.load(stl_path)

    # =========================================================
    # FIX 1: THE COORDINATE FLIP (RAS to LPS)
    # Slicer saves STLs in RAS, but ITK uses LPS. We must flip X and Y.
    # =========================================================
    print("--> Aligning coordinate systems (Flipping RAS to LPS)...")
    mesh.vertices[:, 0] = -mesh.vertices[:, 0]
    mesh.vertices[:, 1] = -mesh.vertices[:, 1]

    # =========================================================
    # FIX 2: WATERTIGHT CHECK
    # Raycasting fails if the mesh has a microscopic hole.
    # =========================================================
    if not mesh.is_watertight:
        print("--> [WARNING] Mesh is not watertight (it has holes).")
        print("--> Wrapping it in a Convex Hull to seal it...")
        # This acts like shrink-wrap to guarantee a sealed, fillable volume
        mesh = mesh.convex_hull

        # Get the voxel dimensions of the original image
    ref_array = sitk.GetArrayFromImage(ref_image)
    z_dim, y_dim, x_dim = ref_array.shape

    print(f"--> Generating voxel grid ({x_dim}x{y_dim}x{z_dim})...")
    z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
    voxel_indices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

    print("--> Translating grid to physical coordinates...")
    spacing = np.array(ref_image.GetSpacing())
    origin = np.array(ref_image.GetOrigin())
    direction = np.array(ref_image.GetDirection()).reshape(3, 3)

    indices_scaled = voxel_indices * spacing
    physical_points = np.dot(indices_scaled, direction.T) + origin
    total_points = len(physical_points)

    # =========================================================
    # THE FIX: BOUNDING BOX OPTIMIZATION
    # =========================================================
    print("--> Filtering points outside the mesh bounding box...")
    min_bound, max_bound = mesh.bounds

    # Fast NumPy check: Is the point inside the general XYZ cube of the mesh?
    in_box_mask = np.all((physical_points >= min_bound) & (physical_points <= max_bound), axis=1)

    # Extract only the points that survived the filter
    points_to_check = physical_points[in_box_mask]
    num_to_check = len(points_to_check)

    print(f"    Reduced workload from {total_points:,} to {num_to_check:,} points!")

    print("--> Raycasting: Checking the filtered points...")
    start_time = time.time()

    # Create an array just for our filtered subset
    mesh_contains_mask = np.zeros(num_to_check, dtype=bool)

    # Smaller, much safer chunk size
    chunk_size = 500_000

    for i in range(0, num_to_check, chunk_size):
        end_idx = min(i + chunk_size, num_to_check)
        percent_done = round((end_idx / num_to_check) * 100, 1)
        print(f"    Processing: {end_idx:,} / {num_to_check:,} ({percent_done}%)")

        mesh_contains_mask[i:end_idx] = mesh.contains(points_to_check[i:end_idx])

    print(f"    Raycasting complete in {round(time.time() - start_time, 1)} seconds.")

    # Map the results back to the master 225M array
    inside_mask = np.zeros(total_points, dtype=bool)
    inside_mask[in_box_mask] = mesh_contains_mask

    print("--> Formatting and saving the final NIfTI mask...")
    # 5. Reshape the flat 1D boolean array back into our 3D image shape
    mask_array = inside_mask.reshape((z_dim, y_dim, x_dim)).astype(np.uint8)

    print("--> Raycasting: Filling the 3D shell (Processing in chunks to save RAM)...")
    start_time = time.time()

    # 4. Create an empty boolean array to hold our final results
    total_points = len(physical_points)
    inside_mask = np.zeros(total_points, dtype=bool)

    # Set chunk size (5 million points per batch is usually safe for most RAM limits)
    chunk_size = 5_000_000

    # Process the points in batches
    for i in range(0, total_points, chunk_size):
        end_idx = min(i + chunk_size, total_points)

        # Optional: Print progress so you know it hasn't frozen
        percent_done = round((end_idx / total_points) * 100, 1)
        print(f"    Processing points: {end_idx:,} / {total_points:,} ({percent_done}%)")

        # Raycast just this chunk and store the results
        inside_mask[i:end_idx] = mesh.contains(physical_points[i:end_idx])

    print(f"    Raycasting complete in {round(time.time() - start_time, 1)} seconds.")

    print("--> Formatting and saving the final NIfTI mask...")
    # 5. Reshape the flat 1D boolean array back into our 3D image shape
    # ... (Keep the rest of your save logic exactly the same)
    mask_array = inside_mask.reshape((z_dim, y_dim, x_dim)).astype(np.uint8)
    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.CopyInformation(ref_image)

    sitk.WriteImage(mask_image, output_path)
    print(f"\n[SUCCESS] Voxelized mask safely saved to: {output_path}")


if __name__ == "__main__":
    my_stl = "prostate_capsule_shell.stl"
    my_original_scan = "bph_model1_08_14_25_0000.nii.gz"
    my_output_mask = "prostate_shell_mask.nii.gz"

    convert_stl_to_nifti(my_stl, my_original_scan, my_output_mask)