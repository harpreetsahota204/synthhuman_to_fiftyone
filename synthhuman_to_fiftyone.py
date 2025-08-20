#!/usr/bin/env python3
import argparse
from pathlib import Path
import fiftyone as fo


def create_synthhuman_dataset(data_dir, dataset_name, overwrite=False):
    data_path = Path(data_dir)
    
    # Find RGB files and assume corresponding depth files exist
    rgb_files = list(data_path.glob("rgb_*.png"))

    dataset = fo.Dataset(
        dataset_name, 
        overwrite=True, 
        persistent=True
        )

    dataset.add_group_field("group", default="rgb")
    
    samples = []

    for rgb_path in rgb_files:
        sample_id = rgb_path.stem.split('_')[1]  # Extract xxxx from rgb_xxxx.png
        mask_path = data_path / f"alpha_{sample_id}.png"
        depth_path = data_path / f"depth_{sample_id}.png"
        normal_path = data_path / f"normal_{sample_id}.png"
        foreground_path = data_path / f"foreground_{sample_id}.png"
        fo3d_path = data_path / f"fo3d_{sample_id}.fo3d"
        
        # Create a group to link different views of the same object
        group = fo.Group()
        
        # Create sample for RGB
        rgb_sample = fo.Sample(
            filepath=rgb_path,
            group=group.element("rgb")
        )
        samples.append(rgb_sample)
        
        # Create sample for alpha mask
        mask_sample = fo.Sample(
            filepath=mask_path,
            group=group.element("mask")
        )
        samples.append(mask_sample)

        # Create sample for foreground
        foreground_sample = fo.Sample(
            filepath=foreground_path,
            group=group.element("foreground")
        )
        samples.append(foreground_sample)
        
        # Create sample for depth
        depth_sample = fo.Sample(
            filepath=depth_path,
            group=group.element("depth")
        )
        samples.append(depth_sample)

        # Create sample for normal
        normal_sample = fo.Sample(
            filepath=normal_path,
            group=group.element("normal")
        )
        samples.append(normal_sample)

        # Create sample for fo3d
        fo3d_sample = fo.Sample(
            filepath=fo3d_path,
            group=group.element("fo3d")
        )
        samples.append(fo3d_sample)

    dataset.add_samples(samples)
    dataset.compute_metadata()
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", required=True)
    parser.add_argument("--dataset_name", "-n", default="SynthHuman")
    
    args = parser.parse_args()
    
    dataset = create_synthhuman_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
    )
    
    print(f"Created dataset '{dataset.name}' with {len(dataset)} samples")
    
if __name__ == "__main__":
    main()