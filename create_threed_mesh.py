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
    """Process a single sample index and save mesh to disk.

    Args:
        idx_str (str): the numeric identifier from filenames like depth_XXXXXX.exr
        input_dir (str | Path): directory containing the input files
    """

    depth_path = Path(input_dir) / f"depth_{idx_str}.exr"
    normal_path = Path(input_dir) / f"normal_{idx_str}.exr"
    alpha_path = Path(input_dir) / f"alpha_{idx_str}.png"
    rgb_path   = Path(input_dir) / f"rgb_{idx_str}.png"
    cam_path   = Path(input_dir) / f"cam_{idx_str}.txt"

    # Skip if mesh already exists
    existing_mesh_path = Path(input_dir) / f"mesh_{idx_str}.ply"
    if existing_mesh_path.exists():
        print(f"Skipping existing mesh: {existing_mesh_path}")
        return

    # load data
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth not found: {depth_path}")
    depth = depth.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None:
        raise FileNotFoundError(f"RGB not found: {rgb_path}")
    if rgb_bgr.ndim == 2:
        rgb_bgr = cv2.cvtColor(rgb_bgr, cv2.COLOR_GRAY2BGR)
    elif rgb_bgr.shape[2] == 4:
        rgb_bgr = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGRA2BGR)
    rgb = rgb_bgr[:, :, ::-1]  # BGR→RGB

    normals_img = cv2.imread(str(normal_path), cv2.IMREAD_UNCHANGED)
    if normals_img is not None:
        normals_img = normals_img.astype(np.float32)
        if normals_img.ndim == 2:
            normals_img = None
        elif normals_img.shape[2] >= 3:
            normals_img = normals_img[:, :, :3]
        else:
            normals_img = None

    intrinsics = load_camera_intrinsics(cam_path)

    # build validity mask: finite depth, positive, not sentinel; then alpha if available
    valid_mask = np.isfinite(depth)
    valid_mask &= depth > 0.0
    valid_mask &= depth != 65504.0

    alpha = _load_alpha_mask(alpha_path, depth.shape)
    if alpha is not None:
        valid_mask &= alpha > 0

    # backproject
    points = depth_to_pointcloud(depth, intrinsics, valid_mask)

    # flatten colors & normals in the same way as points
    mask_flat = valid_mask.flatten()
    colors = rgb.reshape(-1, 3)[mask_flat]

    normals_flat = None
    if normals_img is not None:
        normals_flat = normals_img.reshape(-1, 3)[mask_flat]
        # sanitize normals: finite and non-zero
        finite_mask = np.isfinite(normals_flat).all(axis=1)
        normals_flat = normals_flat[finite_mask]
        points = points[finite_mask]
        colors = colors[finite_mask]
        # normalize to unit length, drop near-zero
        norms = np.linalg.norm(normals_flat, axis=1)
        nonzero = norms > 1e-6
        normals_flat = normals_flat[nonzero]
        points = points[nonzero]
        colors = colors[nonzero]
        normals_flat = normals_flat / np.linalg.norm(normals_flat, axis=1, keepdims=True)

    # build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(points)
    pcd.colors  = o3d.utility.Vector3dVector(colors / 255.0)
    if normals_flat is not None and len(normals_flat) == len(points):
        pcd.normals = o3d.utility.Vector3dVector(normals_flat)
    else:
        # Estimate normals if none available or invalid
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        try:
            pcd.orient_normals_consistent_tangent_plane(20)
        except Exception:
            pass

    # mesh reconstruction (Poisson works best with normals)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    # Remove low-density vertices which often correspond to artifacts
    try:
        densities = np.asarray(densities)
        keep_thresh = np.quantile(densities, 0.02)
        vertices_to_remove = densities < keep_thresh
        mesh.remove_vertices_by_mask(vertices_to_remove)
    except Exception:
        pass
    mesh.compute_vertex_normals()

    # save outputs with matching pattern
    out_ply = depth_path.with_name(depth_path.name.replace("depth_", "mesh_").replace(".exr", ".ply"))
    o3d.io.write_triangle_mesh(str(out_ply), mesh)
    print(f"Saved {out_ply}")

def _process_mesh_item(idx_str, input_dir):
    """Worker function that creates a mesh for the given index and returns output path."""
    create_meshes(idx_str, input_dir)
    return str(Path(input_dir) / f"mesh_{idx_str}.ply")


def process_directory(input_dir, num_workers=None, use_processes=False):
    """
    Iterate over all depth_*.exr files in the directory and generate meshes.
    """
    input_dir = Path(input_dir)
    depth_files = sorted(input_dir.glob("depth_*.exr"))

    if not depth_files:
        print("No depth_*.exr files found.")
        return

    jobs = []
    for depth_path in depth_files:
        idx_str = depth_path.stem.split("_")[1]
        out_ply = input_dir / f"mesh_{idx_str}.ply"
        if out_ply.exists():
            # Skip already processed
            continue
        jobs.append(idx_str)

    if not jobs:
        print("All meshes already exist.")
        return

    ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    max_workers = num_workers or os.cpu_count()

    errors = []
    successes = 0
    with ExecutorClass(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_mesh_item, idx_str, str(input_dir)) for idx_str in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Meshing", unit="file"):
            try:
                _ = future.result()
                successes += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))

    print(f"Meshed {successes}/{len(jobs)} items.")
    if errors:
        print("Encountered errors (showing up to 10):")
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
