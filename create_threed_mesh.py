import os
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import fiftyone as fo
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.ndimage import binary_erosion

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def load_camera_intrinsics(path_txt):
    """Load intrinsics from cam_XXXX.txt into a 3x3 numpy array."""
    return np.loadtxt(path_txt)

def depth_to_pointcloud(depth, intrinsics, mask=None):
    """Backproject depth map into a point cloud."""
    h, w = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts = np.stack((x, y, z), axis=-1)

    if mask is not None:
        pts = pts[mask > 0]

    return pts.reshape(-1, 3)


def upsample_pointcloud(points, colors, normals=None, method="radius", radius=None, k_neighbors=8, fast_mode=False):
    """
    Upsample/densify a point cloud to improve mesh reconstruction quality.
    
    Args:
        points (np.ndarray): Nx3 array of 3D points
        colors (np.ndarray): Nx3 array of RGB colors
        normals (np.ndarray, optional): Nx3 array of normals
        method (str): "radius" for radius-based or "knn" for k-nearest neighbors
        radius (float, optional): radius for upsampling (if None, uses avg distance)
        k_neighbors (int): number of neighbors for interpolation
        fast_mode (bool): if True, use lighter upsampling (sparse sampling)
        
    Returns:
        tuple: (upsampled_points, upsampled_colors, upsampled_normals)
    """
    if len(points) < 3:
        return points, colors, normals
    
    # Fast mode: only upsample sparse regions, skip if already dense
    if fast_mode and len(points) > 5000:
        return points, colors, normals
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Calculate average distance if radius not provided
    if radius is None:
        distances = pcd.compute_nearest_neighbor_distance()
        radius = np.mean(distances) * (1.2 if fast_mode else 0.8)
    
    # Build KDTree for efficient neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    new_points = []
    new_colors = []
    new_normals = [] if normals is not None else None
    
    # Add original points
    new_points.extend(points)
    new_colors.extend(colors)
    if normals is not None:
        new_normals.extend(normals)
    
    # Fast mode: sample every Nth point for interpolation
    step_size = 3 if fast_mode else 1
    max_neighbors = 2 if fast_mode else 3
    
    # Generate interpolated points between neighbors
    for i in range(0, len(points), step_size):
        point = points[i]
        if method == "radius":
            [k, idx, _] = kdtree.search_radius_vector_3d(point, radius)
        else:  # knn
            [k, idx, _] = kdtree.search_knn_vector_3d(point, k_neighbors)
        
        if k > 1:
            neighbor_indices = idx[1:min(max_neighbors + 1, k)]
            for j in neighbor_indices:
                # Interpolate between current point and neighbor
                interp_point = 0.5 * (points[i] + points[j])
                interp_color = 0.5 * (colors[i] + colors[j])
                
                new_points.append(interp_point)
                new_colors.append(interp_color)
                
                if normals is not None:
                    interp_normal = 0.5 * (normals[i] + normals[j])
                    # Normalize the interpolated normal
                    norm = np.linalg.norm(interp_normal)
                    if norm > 1e-6:
                        interp_normal = interp_normal / norm
                    new_normals.append(interp_normal)
    
    return (np.array(new_points), 
            np.array(new_colors), 
            np.array(new_normals) if new_normals else None)


def _load_alpha_mask(alpha_path: Path, expected_shape):
    """Load alpha as a single-channel binary mask matching expected HxW."""
    if not alpha_path.exists():
        return None

    alpha = cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED)
    if alpha is None:
        return None

    if alpha.ndim == 3:
        alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)

    alpha = (alpha > 0).astype(np.uint8)
    if alpha.shape != expected_shape[:2]:
        alpha = cv2.resize(alpha, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_NEAREST)
    return alpha

def create_meshes(idx_str, input_dir, enable_upsampling=True, enable_smoothing=True, fast_mode=False, skip_outlier_removal=False):
    """
    Reconstruct a mesh from a single dataset sample using Ball Pivoting.

    Args:
        idx_str (str): zero-padded numeric identifier (e.g. "000123")
        input_dir (str | Path): directory containing the dataset files
        enable_upsampling (bool): whether to upsample point cloud for density
        enable_smoothing (bool): whether to apply mesh smoothing postprocessing

    Workflow:
        1. Load depth, RGB, normals, and intrinsics.
        2. Backproject depth into a point cloud.
        3. Apply morphological erosion to alpha mask to reduce edge artifacts.
        4. Optionally upsample point cloud for better density.
        5. Apply statistical outlier removal.
        6. Attach RGB colors and (if valid) normals.
        7. Reconstruct a mesh via Ball Pivoting (BPA) with adaptive radii.
        8. Apply mesh postprocessing (cleanup, smoothing).
        9. Flip Y-axis to make geometry Y-up, fix winding, save to PLY.
    """
    input_dir = Path(input_dir)

    # File paths
    depth_path = input_dir / f"depth_{idx_str}.exr"
    normal_path = input_dir / f"normal_{idx_str}.exr"
    alpha_path = input_dir / f"alpha_{idx_str}.png"
    rgb_path   = input_dir / f"rgb_{idx_str}.png"
    cam_path   = input_dir / f"cam_{idx_str}.txt"
    out_ply    = input_dir / f"mesh_{idx_str}.ply"

    # Skip if mesh already exists
    if out_ply.exists():
        print(f"Skipping existing mesh: {out_ply}")
        return

    # ---------------------------
    # Load depth + intrinsics
    # ---------------------------
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth not found: {depth_path}")
    depth = depth.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]  # keep first channel if multi-channel EXR

    intrinsics = load_camera_intrinsics(cam_path)

    # ---------------------------
    # Load RGB
    # ---------------------------
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None:
        raise FileNotFoundError(f"RGB not found: {rgb_path}")
    if rgb_bgr.ndim == 2:
        rgb_bgr = cv2.cvtColor(rgb_bgr, cv2.COLOR_GRAY2BGR)
    elif rgb_bgr.shape[2] == 4:
        rgb_bgr = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGRA2BGR)
    rgb = rgb_bgr[:, :, ::-1]  # BGR → RGB

    # ---------------------------
    # Load normals (optional)
    # ---------------------------
    normals_img = cv2.imread(str(normal_path), cv2.IMREAD_UNCHANGED)
    if normals_img is not None:
        normals_img = normals_img.astype(np.float32)
        if normals_img.ndim == 2:
            normals_img = None
        elif normals_img.shape[2] >= 3:
            normals_img = normals_img[:, :, :3]
            # If encoded [0,1], map → [-1,1]
            if 0.0 <= float(np.nanmin(normals_img)) and float(np.nanmax(normals_img)) <= 1.0:
                normals_img = normals_img * 2.0 - 1.0
        else:
            normals_img = None

    # ---------------------------
    # Build validity mask
    # ---------------------------
    valid_mask = np.isfinite(depth)
    valid_mask &= depth > 0.0
    valid_mask &= depth != 65504.0  # sentinel in EXR

    alpha = _load_alpha_mask(alpha_path, depth.shape)
    if alpha is not None:
        # Apply morphological erosion to reduce edge artifacts
        alpha_eroded = binary_erosion(alpha > 0, structure=np.ones((3, 3)), iterations=1)
        valid_mask &= alpha_eroded

    # ---------------------------
    # Backproject to 3D points
    # ---------------------------
    points = depth_to_pointcloud(depth, intrinsics, valid_mask)
    mask_flat = valid_mask.flatten()
    colors = rgb.reshape(-1, 3)[mask_flat]

    normals_flat = None
    if normals_img is not None:
        normals_candidate = normals_img.reshape(-1, 3)[mask_flat]
        finite_mask = np.isfinite(normals_candidate).all(axis=1)
        norms = np.linalg.norm(normals_candidate[finite_mask], axis=1)
        nonzero_mask = norms > 1e-6
        if np.any(nonzero_mask):
            valid_indices = np.where(finite_mask)[0][nonzero_mask]
            points = points[valid_indices]
            colors = colors[valid_indices]
            normals_flat = normals_candidate[finite_mask][nonzero_mask]
            normals_flat = normals_flat / np.linalg.norm(normals_flat, axis=1, keepdims=True)

    # ---------------------------
    # Upsample point cloud for better mesh quality
    # ---------------------------
    if enable_upsampling and len(points) > 100:  # Only upsample if we have sufficient points
        points, colors, normals_flat = upsample_pointcloud(points, colors, normals_flat, fast_mode=fast_mode)
        print(f"Upsampled point cloud to {len(points)} points")

    # ---------------------------
    # Build Open3D point cloud
    # ---------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    if normals_flat is not None and len(normals_flat) == len(points):
        pcd.normals = o3d.utility.Vector3dVector(normals_flat)
    else:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_consistent_tangent_plane(20)

    # ---------------------------
    # Statistical outlier removal (skip for clean synthetic data)
    # ---------------------------
    if not skip_outlier_removal and len(pcd.points) > 50 and not fast_mode:
        # Reduce neighbors for speed, or skip if point cloud is large and clean
        nb_neighbors = 8 if len(pcd.points) > 10000 else 15
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=2.0)
        removed_count = len(pcd.points) - len(ind)
        
        # Skip outlier removal if very few outliers found (clean synthetic data)
        if removed_count > len(pcd.points) * 0.001:  # Only apply if >0.1% outliers
            pcd = pcd.select_by_index(ind)
            print(f"Removed {removed_count} outlier points")
        else:
            print(f"Skipped outlier removal - data appears clean ({removed_count} outliers found)")
    elif skip_outlier_removal:
        print("Outlier removal disabled for synthetic data")

    # ---------------------------
    # Ball Pivoting Reconstruction
    # ---------------------------
    # Estimate average spacing between points
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    # Adaptive radii: use fewer radii for speed, more for quality
    if len(pcd.points) > 15000:  # Large point clouds: use fewer radii for speed
        radii = o3d.utility.DoubleVector([
            avg_dist * 1.5,    # Medium gaps
            avg_dist * 2.5,    # Larger gaps
        ])
    elif len(pcd.points) > 8000:  # Medium point clouds: balanced approach
        radii = o3d.utility.DoubleVector([
            avg_dist * 1.25,   # Small gaps
            avg_dist * 2.0,    # Larger gaps
            avg_dist * 2.8,    # Large holes
        ])
    else:  # Small point clouds: use more radii for quality
        radii = o3d.utility.DoubleVector([
            avg_dist * 1.0,    # Fine details
            avg_dist * 1.5,    # Medium gaps
            avg_dist * 2.5,    # Larger gaps
        ])

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, radii
    )
    mesh.compute_vertex_normals()

    # ---------------------------
    # Mesh postprocessing for quality improvement
    # ---------------------------
    # Remove degenerate and duplicate triangles
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    
    # Apply Laplacian smoothing for visual smoothness
    if enable_smoothing and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        # Adaptive smoothing: fewer iterations for large meshes
        if len(mesh.vertices) > 20000:
            iterations = 2  # Light smoothing for large meshes
        elif len(mesh.vertices) > 10000:
            iterations = 3  # Medium smoothing
        else:
            iterations = 5  # Full smoothing for small meshes
            
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
        mesh.compute_vertex_normals()
        print(f"Applied {iterations} iterations of Laplacian smoothing to mesh with {len(mesh.vertices)} vertices")

    # ---------------------------
    # Flip Y-axis → make geometry Y-up
    # ---------------------------
    flip_Y = np.eye(4)
    flip_Y[1, 1] = -1.0
    mesh.transform(flip_Y)

    # Fix triangle winding after flip
    tris = np.asarray(mesh.triangles)
    if tris.size > 0:
        mesh.triangles = o3d.utility.Vector3iVector(tris[:, [0, 2, 1]])
    mesh.compute_vertex_normals()

    # ---------------------------
    # Save mesh
    # ---------------------------
    o3d.io.write_triangle_mesh(str(out_ply), mesh)
    print(f"Saved {out_ply}")


def _process_mesh_item(args):
    """Worker function that creates a mesh for the given index and returns output path."""
    idx_str, input_dir, enable_upsampling, enable_smoothing, fast_mode, skip_outlier_removal = args
    create_meshes(idx_str, input_dir, enable_upsampling, enable_smoothing, fast_mode, skip_outlier_removal)
    return str(Path(input_dir) / f"mesh_{idx_str}.ply")

def process_directory(input_dir, num_workers=None, use_processes=False, enable_upsampling=True, enable_smoothing=True, fast_mode=False, skip_outlier_removal=False):
    """
    Generate meshes for all depth_*.exr files in a dataset directory using Ball Pivoting.

    Args:
        input_dir (str | Path): directory containing the dataset files
        num_workers (int, optional): number of parallel workers (default: CPU count)
        use_processes (bool): whether to use process-based parallelism
        enable_upsampling (bool): whether to upsample point cloud for density
        enable_smoothing (bool): whether to apply mesh smoothing postprocessing

    Workflow:
        1. Find all depth_*.exr files.
        2. Skip samples with an existing mesh_*.ply.
        3. Reconstruct meshes in parallel with workers using improved BPA pipeline.
    """
    input_dir = Path(input_dir)
    depth_files = sorted(input_dir.glob("depth_*.exr"))

    if not depth_files:
        print("No depth_*.exr files found.")
        return

    # Collect jobs for missing meshes
    jobs = []
    for depth_path in depth_files:
        idx_str = depth_path.stem.split("_")[1]
        out_ply = input_dir / f"mesh_{idx_str}.ply"
        if not out_ply.exists():
            jobs.append(idx_str)

    if not jobs:
        print("All meshes already exist.")
        return

    ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    max_workers = num_workers or os.cpu_count()

    errors = []
    successes = 0
    with ExecutorClass(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_mesh_item, (idx_str, str(input_dir), enable_upsampling, enable_smoothing, fast_mode, skip_outlier_removal))
                   for idx_str in jobs]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Meshing", unit="file"):
            try:
                _ = future.result()
                successes += 1
            except Exception as exc:
                errors.append(str(exc))

    print(f"✅ Meshed {successes}/{len(jobs)} items.")
    if errors:
        print("⚠️ Encountered errors (showing up to 10):")
        for msg in errors[:10]:
            print(" -", msg)


def create_fo3d(input_dir):
    """
    Process a mesh and create a FiftyOne 3D scene.
    
    This function takes a mesh, creates a 3D scene with appropriate camera 
    settings, adds the mesh, and saves the scene in the same directory.
    
    Args:
        input_dir (str): directory containing the input meshes
        
    Returns:
        None. The function writes the scene file to disk and prints status.
    """
    input_dir = Path(input_dir)
    mesh_files = sorted(input_dir.glob("mesh_*.ply"))
    if not mesh_files:
        print("No mesh_*.ply files found.")
        return
    
    for mesh_path in mesh_files:
        # Create a new FiftyOne 3D scene per mesh
        scene = fo.Scene()
        scene.camera = fo.PerspectiveCamera(up="Y")

        material = fo.PointCloudMaterial(
            shading_mode="rgb",
            attenuate_by_distance=True)

        abs_mesh_path = os.path.abspath(mesh_path)
        name_parts = Path(abs_mesh_path).stem.split('_')
        mesh_id = name_parts[1] if len(name_parts) > 1 else Path(abs_mesh_path).stem

        # Skip if .fo3d already exists
        output_path = input_dir / f"fo3d_{mesh_id}.fo3d"
        if output_path.exists():
            print(f"Skipping existing fo3d: {output_path}")
            continue

        # Create a mesh from the point cloud with RGB coloring
        mesh = fo.PlyMesh(
            "mesh",
            abs_mesh_path,
            is_point_cloud=False,
            default_material=material,
            center_geometry=True,
        )
        # Add the mesh to our scene
        scene.add(mesh)

        # Save the scene as a .fo3d file in the same directory
        scene.write(str(output_path), resolve_relative_paths=True)
        print(f"Processed: {abs_mesh_path} -> {output_path}")

def main(input_dir, workers=None, processes=False, enable_upsampling=True, enable_smoothing=True, fast_mode=False, skip_outlier_removal=False):
    process_directory(input_dir, num_workers=workers, use_processes=processes, 
                     enable_upsampling=enable_upsampling, enable_smoothing=enable_smoothing, 
                     fast_mode=fast_mode, skip_outlier_removal=skip_outlier_removal)
    create_fo3d(input_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create 3D meshes from a SynthHuman-like dataset directory with improved BPA pipeline")
    parser.add_argument("input_dir", help="Path to the input dataset directory")
    parser.add_argument("-j", "--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--processes", action="store_true", help="Use process-based parallelism (default: threads)")
    parser.add_argument("--no-upsampling", action="store_true", help="Disable point cloud upsampling")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable mesh Laplacian smoothing")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode (reduced quality for speed)")
    parser.add_argument("--skip-outliers", action="store_true", help="Skip outlier removal (recommended for clean synthetic data)")
    args = parser.parse_args()

    main(args.input_dir, workers=args.workers, processes=args.processes,
         enable_upsampling=not args.no_upsampling, enable_smoothing=not args.no_smoothing, 
         fast_mode=args.fast, skip_outlier_removal=args.skip_outliers)
