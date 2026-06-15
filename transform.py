import numpy as np
import open3d as o3d


def align_and_generate_shell(file_paths, output_path, poisson_depth=8):
    """Micro-aligns multiple STL files using ICP, merges them,

    and uses Poisson surface reconstruction to create a tight, watertight shell.
    """
    if not file_paths:
        print("No files provided.")
        return

    print("--> Loading baseline mesh (Anchor)...")
    # Load the first mesh as the fixed geometric anchor
    anchor_mesh = o3d.io.read_triangle_mesh(file_paths[0])

    # Convert anchor to a dense point cloud for ICP matching
    # Sampling points ensures dense surface coverage even across large triangles
    anchor_pcd = anchor_mesh.sample_points_uniformly(number_of_points=50000)

    # Master list to hold all aligned points
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd += anchor_pcd

    # 1. Surface-to-Surface Micro-Alignment (ICP)
    # -------------------------------------------------------------
    # Max distance the algorithm looks for matching points (adjust based on your scale)
    # For prostate meshes in millimeter scale, 2.0 to 5.0 mm is a good search radius
    threshold = 3.0
    trans_init = np.identity(4)  # Start with Slicer's baseline alignment

    for path in file_paths[1:]:
        print(f"--> Aligning {path} to anchor...")
        moving_mesh = o3d.io.read_triangle_mesh(path)
        moving_pcd = moving_mesh.sample_points_uniformly(
            number_of_points=50000
        )

        # Run Point-to-Plane or Point-to-Point ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            moving_pcd,
            anchor_pcd,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        # Apply the optimized transformation matrix to the moving point cloud
        moving_pcd.transform(reg_p2p.transformation)
        combined_pcd += moving_pcd

    print(
        f"--> Merged point cloud contains {len(combined_pcd.points)} total points."
    )

    # 2. Watertight Shell Generation (Poisson Reconstruction)
    # -------------------------------------------------------------
    print("--> Computing surface normals...")
    combined_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=5.0, max_nn=30
        )
    )
    # Ensure all normals point outward consistently
    combined_pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0.0, 0.0, 0.0])
    )
    # Invert if the camera calculation accidentally targets the interior center
    combined_pcd.normals = o3d.utility.Vector3dVector(
        -np.asarray(combined_pcd.normals)
    )

    print(f"--> Running Poisson Surface Reconstruction (Depth={poisson_depth})...")
    # depth: Higher values (9-10) capture highly specific details but can fit to noise.
    # A depth of 7-8 is ideal for a clean, tight, organic "capsule" boundary.
    shell_mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            combined_pcd, depth=poisson_depth
        )
    )

    # Clean up low-density artifacts
    print("--> Trimming low-density boundary artifacts...")
    vertices_to_remove = densities < np.percentile(np.asarray(densities), 5)
    shell_mesh.remove_vertices_by_mask(vertices_to_remove)

    print("--> Applying Laplacian smoothing...")
    # You can increase iterations (e.g., 10 to 30) for a smoother finish
    shell_mesh = shell_mesh.filter_smooth_laplacian(number_of_iterations=15)
    shell_mesh.compute_vertex_normals()  # Recompute lighting for the smoothed shape

    # 2. Compute lighting vectors so it isn't a black blob
    shell_mesh.compute_vertex_normals()
    # -----------------------------

    # 3. Export the finished shell
    # -------------------------------------------------------------
    o3d.io.write_triangle_mesh(output_path, shell_mesh)
    print(f"[SUCCESS] Watertight capsule shell saved to: {output_path}")

    # Optional Visualizer to check your work
    o3d.visualization.draw_geometries(
        [shell_mesh], mesh_show_back_face=True, window_name="Capsule Shell"
    )


# Example usage:
if __name__ == "__main__":
    # Add your exported, hardened STL paths here
    # Keep the best/most complete file as the first entry (Anchor)
    my_stls = ["C:/Users/iven0/OneDrive/Desktop/VINE-Lab-Software/BPH_capsules/bph_model1_08_14_25_capsule.stl", "C:/Users/iven0/OneDrive/Desktop/VINE-Lab-Software/BPH_capsules/transformed_bph_model2_09_17_25.stl"]

    align_and_generate_shell(
        file_paths=my_stls,
        output_path="prostate_capsule_shell_test.stl",
        poisson_depth=8,  # Change to 7 for a smoother wrap, 9 for tighter details
    )