import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def load_alpha(alpha_path, shape):
    """Load alpha mask (1=foreground, 0=background), resize if necessary."""
    if not os.path.exists(alpha_path):
        return None

    alpha = cv2.imread(alpha_path, cv2.IMREAD_UNCHANGED)
    if alpha is None:
        return None

    if alpha.ndim == 3:
        alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)

    alpha = (alpha > 0).astype(np.uint8)

    if alpha.shape != shape[:2]:
        alpha = cv2.resize(alpha, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    return alpha


def save_depth_heatmap(depth_path, alpha, out_path, foreground_out_path=None):
    """
    Convert depth EXR → RGBA PNG heatmap using official conventions:
    - Foreground only
    - Ignore sentinel depth values (65504)
    - Invert: closer = warmer
    - Apply Inferno colormap
    """
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]

    # Apply alpha mask and remove sentinel
    if alpha is not None:
        foreground = np.logical_and(alpha > 0, depth != 65504)
    else:
        foreground = depth != 65504

    if not np.any(foreground):
        # Nothing to visualize; output a transparent image
        h, w = depth.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        cv2.imwrite(out_path, rgba)
        return

    # Prepare array for normalized values (avoid NaNs in background)
    norm = np.zeros_like(depth, dtype=np.float32)

    # Normalize only on foreground to [0,1]
    fg_values = depth[foreground]
    min_val = float(np.min(fg_values))
    max_val = float(np.max(fg_values))
    if max_val != min_val:
        norm_fg = (fg_values - min_val) / (max_val - min_val)
    else:
        norm_fg = np.zeros_like(fg_values, dtype=np.float32)
    norm[foreground] = norm_fg

    # Invert: closer = warmer
    norm = 1.0 - np.clip(norm, 0.0, 1.0)
    norm_uint8 = (norm * 255.0).astype(np.uint8)

    # Apply Inferno colormap
    rgb = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_INFERNO)

    # Alpha channel
    if alpha is not None:
        a = (alpha * 255).astype(np.uint8)
    else:
        a = np.full(depth.shape[:2], 255, np.uint8)

    rgba = np.dstack([rgb, a])
    cv2.imwrite(out_path, rgba)


def save_normal_heatmap(normal_path, alpha, out_path, foreground_out_path=None):
    """
    Convert normal EXR → RGBA PNG using official conventions:
    - Normalize [-1,1] → [0,255]
    - Convert RGB→BGR
    - Mask background with alpha
    """
    normals = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if normals.shape[2] >= 3:
        normals = normals[:, :, :3]

    rgb = ((normals * 0.5 + 0.5) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Alpha channel
    if alpha is not None:
        a = (alpha * 255).astype(np.uint8)
        rgb[alpha == 0] = 0
    else:
        a = np.full(normals.shape[:2], 255, np.uint8)

    rgba = np.dstack([rgb, a])
    cv2.imwrite(out_path, rgba)


def save_rgb_foreground(rgb_path, alpha, out_path):
    """
    Save RGB with background removed using alpha. Output is same size as input,
    with original colors preserved and background fully transparent (BGRA PNG).
    """
    if alpha is None:
        return

    rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    if rgb is None:
        return

    # Normalize to 3-channel BGR
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    elif rgb.shape[2] == 4:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)

    # Ensure alpha matches RGB size
    if alpha.shape != rgb.shape[:2]:
        alpha_resized = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        alpha_resized = alpha

    a = (alpha_resized * 255).astype(np.uint8)
    rgba = np.dstack([rgb, a])
    # Optional: explicitly zero background color to avoid matte fringe
    rgba[alpha_resized == 0, :3] = 0
    cv2.imwrite(out_path, rgba)


def _process_item(depth_path, normal_path, alpha_path, rgb_path, depth_out, normal_out, fg_rgb_out):
    """Worker function to process a single item. Top-level to be picklable."""
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise FileNotFoundError(f"Failed to read depth EXR: {depth_path}")

    alpha_for_depth = load_alpha(alpha_path, depth_img.shape)

    # Depth
    save_depth_heatmap(depth_path, alpha_for_depth, depth_out)

    # Normal (optional)
    if os.path.exists(normal_path):
        save_normal_heatmap(normal_path, alpha_for_depth, normal_out)
    else:
        # If missing, still create a transparent placeholder to keep outputs consistent
        h, w = depth_img.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        cv2.imwrite(normal_out, rgba)

    # Foreground RGB (optional)
    if os.path.exists(rgb_path):
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        if rgb_img is not None:
            alpha_for_rgb = load_alpha(alpha_path, rgb_img.shape)
            save_rgb_foreground(rgb_path, alpha_for_rgb, fg_rgb_out)

    return depth_out, normal_out


def process_directory(in_dir, num_workers=None, use_processes=False):
    """
    Iterate over all depth_XXXXXX.exr files:
    - Find matching normal and alpha
    - Save depth_XXXXXX.png and normal_XXXXXX.png with transparency
    """
    depth_files = [
        fname for fname in os.listdir(in_dir)
        if fname.startswith("depth_") and fname.endswith(".exr")
    ]

    jobs = []
    for fname in depth_files:
        stem_num = fname.split("_")[1].split(".")[0]

        depth_path = os.path.join(in_dir, f"depth_{stem_num}.exr")
        normal_path = os.path.join(in_dir, f"normal_{stem_num}.exr")
        alpha_path = os.path.join(in_dir, f"alpha_{stem_num}.png")
        rgb_path = os.path.join(in_dir, f"rgb_{stem_num}.png")

        depth_out = os.path.join(in_dir, f"depth_{stem_num}.png")
        normal_out = os.path.join(in_dir, f"normal_{stem_num}.png")
        fg_rgb_out = os.path.join(in_dir, f"foreground_{stem_num}.png")

        jobs.append((depth_path, normal_path, alpha_path, rgb_path, depth_out, normal_out, fg_rgb_out))

    if not jobs:
        print("No depth EXR files found.")
        return

    ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    max_workers = num_workers or os.cpu_count()

    errors = []
    successes = 0
    with ExecutorClass(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_item, *job)
            for job in jobs
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting", unit="file"):
            try:
                depth_out, normal_out = future.result()
                successes += 1
            except Exception as exc:  # noqa: BLE001 - want to capture and summarize
                errors.append(str(exc))

    print(f"Converted {successes}/{len(jobs)} items.")
    if errors:
        print("Encountered errors (showing up to 10):")
        for msg in errors[:10]:
            print(" -", msg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="Input directory with EXRs and alphas")
    parser.add_argument("-j", "--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--processes", action="store_true", help="Use process-based parallelism (default: threads)")
    args = parser.parse_args()
    process_directory(args.indir, num_workers=args.workers, use_processes=args.processes)