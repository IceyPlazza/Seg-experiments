import numpy as np
import open3d as o3d
import argparse
import sys


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
    combined_pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0.0, 0.0, 0.0])
    )
    combined_pcd.normals = o3d.utility.Vector3dVector(
        -np.asarray(combined_pcd.normals)
    )

    print(f"--> Running Poisson Surface Reconstruction (Depth={poisson_depth})...")
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
    shell_mesh = shell_mesh.filter_smooth_laplacian(number_of_iterations=15)

    # Compute lighting vectors so it isn't a black blob
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


# ==========================================
# CLI Execution Block
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align multiple STL meshes and generate a continuous Poisson shell."
    )

    # The nargs='+' turns all inputs after this flag into a list of strings
    parser.add_argument(
        "-i", "--inputs",
        nargs='+',
        required=True,
        help="List of input STL files separated by spaces. The FIRST file acts as the anchor."
    )

    parser.add_argument(
        "-o", "--output",
        default="prostate_capsule_shell.stl",
        help="Path for the output STL [Default: prostate_capsule_shell.stl]"
    )

    parser.add_argument(
        "-d", "--depth",
        type=int,
        default=8,
        help="Poisson reconstruction depth (e.g., 7 for smooth, 9 for detailed) [Default: 8]"
    )

    args = parser.parse_args()

    try:
        # Pass the parsed arguments into your existing function
        align_and_generate_shell(
            file_paths=args.inputs,
            output_path=args.output,
            poisson_depth=args.depth
        )
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)