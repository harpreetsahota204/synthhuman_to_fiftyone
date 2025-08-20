import os
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import fiftyone as fo
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

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

def create_meshes(idx_str, input_dir):
    """
    Reconstruct a mesh from a single dataset sample using Ball Pivoting.

    Args:
        idx_str (str): zero-padded numeric identifier (e.g. "000123")
        input_dir (str | Path): directory containing the dataset files

    Workflow:
        1. Load depth, RGB, normals, and intrinsics.
        2. Backproject depth into a point cloud.
        3. Attach RGB colors and (if valid) normals.
        4. Reconstruct a mesh via Ball Pivoting (BPA).
        5. Flip Y-axis to make geometry Y-up, fix winding, save to PLY.
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
        valid_mask &= alpha > 0

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
    # Ball Pivoting Reconstruction
    # ---------------------------
    # Estimate average spacing between points
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    # Use multiple radii as multiples of avg spacing
    radii = o3d.utility.DoubleVector([avg_dist * 1.5,
                                      avg_dist * 2.5,
                                      avg_dist * 3.0])

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, radii
    )
    mesh.compute_vertex_normals()

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


def _process_mesh_item(idx_str, input_dir):
    """Worker function that creates a mesh for the given index and returns output path."""
    create_meshes(idx_str, input_dir)
    return str(Path(input_dir) / f"mesh_{idx_str}.ply")

def process_directory(input_dir, num_workers=None, use_processes=False):
    """
    Generate meshes for all depth_*.exr files in a dataset directory using Ball Pivoting.

    Args:
        input_dir (str | Path): directory containing the dataset files
        num_workers (int, optional): number of parallel workers (default: CPU count)
        use_processes (bool): whether to use process-based parallelism

    Workflow:
        1. Find all depth_*.exr files.
        2. Skip samples with an existing mesh_*.ply.
        3. Reconstruct meshes in parallel with workers.
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
        futures = [executor.submit(create_meshes, idx_str, str(input_dir))
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
        output_path = input_dir / f"{mesh_id}.fo3d"
        if output_path.exists():
            print(f"Skipping existing fo3d: {output_path}")
            continue

        # Create a mesh from the point cloud with RGB coloring
        mesh = fo.PlyMesh(
            "mesh",
            abs_mesh_path,
            is_point_cloud=False,
            default_material=material,
            center_geometry=False,
        )
        # Add the mesh to our scene
        scene.add(mesh)

        # Save the scene as a .fo3d file in the same directory
        scene.write(str(output_path))
        print(f"Processed: {abs_mesh_path} -> {output_path}")

def main(input_dir, workers=None, processes=False):
    process_directory(input_dir, num_workers=workers, use_processes=processes)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create 3D meshes from a SynthHuman-like dataset directory")
    parser.add_argument("input_dir", help="Path to the input dataset directory")
    parser.add_argument("-j", "--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--processes", action="store_true", help="Use process-based parallelism (default: threads)")
    args = parser.parse_args()

    main(args.input_dir, workers=args.workers, processes=args.processes)
