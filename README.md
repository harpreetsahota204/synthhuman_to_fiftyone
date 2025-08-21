# SynthHuman to FiftyOne Pipeline

This repository provides a pipeline for processing the SynthHuman dataset into a format compatible with FiftyOne, enabling easy visualization and exploration of synthetic human data.

## Overview

The SynthHuman dataset contains synthetic human images with various data layers (RGB, depth, normal maps, alpha masks). This pipeline:

1. Downloads the SynthHuman dataset
2. Optionally prunes the dataset to a manageable size
3. Converts EXR depth and normal maps to viewable PNG format
4. Creates 3D meshes from the depth and normal data
5. Imports everything into FiftyOne for visualization and exploration

## Requirements

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies for OpenEXR (Ubuntu/Debian)
sudo apt-get install libopenexr-dev openexr
```

Key dependencies:
- FiftyOne: For dataset visualization and management
- OpenCV (with OpenEXR support): For image processing and EXR handling
- Open3D: For 3D mesh generation
- numpy: For numerical operations
- tqdm: For progress tracking
- OpenEXR: For reading/writing high-precision depth and normal maps

## Environment Variables and System Dependencies

The following environment variables are automatically set in the scripts:
- `OPENCV_IO_ENABLE_OPENEXR=1`: Required for OpenCV to read/write OpenEXR files

**System Requirements for OpenEXR Support:**
```bash
# Ubuntu/Debian
sudo apt-get install libopenexr-dev openexr

# macOS
brew install openexr

# Windows
# Install via conda: conda install -c conda-forge openexr
```

These system packages are required for the Python OpenEXR bindings to work properly.

## Pipeline Steps

### 1. Download SynthHuman Dataset

```bash
# Make sure you're in the synthhuman_to_fiftyone directory
cd synthhuman_to_fiftyone

# Download a sample (for testing)
python download_data.py ./ sample

# Download specific chunks (0-59)
python download_data.py ./ 0 20 40

# Download a range of chunks
# 0-19:  Face samples
# 20-39: Full body samples  
# 40-59: Upper body samples
```

This step:
- Downloads the specified chunks from the SynthHuman dataset from Microsoft servers
- Extracts the ZIP files into the SynthHuman subdirectory
- Cleans up temporary files after extraction

Note: All scripts expect the SynthHuman directory to be a subdirectory within synthhuman_to_fiftyone.

### 2. (Optional) Prune Dataset

```bash
# Run the pruning script
python prune_dataset.py
```

This step:
- Identifies all unique IDs in the dataset from the SynthHuman subdirectory
- Removes a specified percentage (default 80%) of the data
- Performs a dry run first for safety
- Asks for confirmation before actual deletion

Useful for reducing the dataset size for testing or when storage is limited.

### 3. Convert Images

```bash
# Run the image converter
python image_converter.py ./SynthHuman -j 8
```

Options:
- `-j/--workers`: Number of parallel workers (default: CPU count)
- `--processes`: Use process-based parallelism instead of threads

This step:
- Converts depth EXR files to PNG heatmaps with transparency
- Converts normal EXR files to PNG with correct RGB encoding
- Creates foreground RGB images with transparent backgrounds
- Processes files in parallel for speed

### 4. Create 3D Meshes

```bash
# Run the 3D mesh creation script
python create_threed_mesh.py ./SynthHuman -j 8
```

Options:
- `-j/--workers`: Number of parallel workers (default: CPU count)
- `--processes`: Use process-based parallelism instead of threads

This step:
- Backprojects depth maps into 3D point clouds
- Uses camera intrinsics to ensure correct geometry
- Applies Ball Pivoting Algorithm to reconstruct meshes
- Creates FiftyOne 3D scenes (.fo3d files) for each mesh
- Processes files in parallel for speed

### 5. Import to FiftyOne

```bash
# Run the FiftyOne import script
python synthhuman_to_fiftyone.py -d ./SynthHuman -n SynthHuman
```

Options:
- `-d/--data_dir`: Path to the processed SynthHuman directory
- `-n/--dataset_name`: Name for the FiftyOne dataset (default: "SynthHuman")

This step:
- Creates a FiftyOne dataset with all processed files
- Organizes files into groups (RGB, mask, depth, normal, foreground, 3D)
- Enables navigation between different views of the same subject

## Viewing the Dataset

After completing the pipeline, you can view the dataset in FiftyOne:

```python
import fiftyone as fo

# Load the dataset
dataset = fo.load_dataset("SynthHuman")

# Launch the FiftyOne App
session = fo.launch_app(dataset)
```

## Technical Notes

- **EXR Handling**: OpenEXR files are used for high-precision depth and normal maps. OpenCV requires special configuration to handle these files.

- **Mesh Generation**: The Ball Pivoting Algorithm is used to create meshes from point clouds. Multiple radii are used to capture details at different scales.

- **Parallelism**: Most processing steps support parallel execution for faster processing.

- **Memory Usage**: Creating meshes can be memory-intensive. If you encounter memory issues, reduce the number of parallel workers.

## Licenses

- SynthHuman Dataset: [CDLA-2.0 license](https://cdla.dev/permissive-2-0/)
- Code in this repository: MIT License

# Citation

```bibtex
@misc{saleh2025david,
    title={{DAViD}: Data-efficient and Accurate Vision Models from Synthetic Data},
    author={Fatemeh Saleh and Sadegh Aliakbarian and Charlie Hewitt and Lohit Petikam and Xiao-Xian and Antonio Criminisi and Thomas J. Cashman and Tadas Baltrušaitis},
    year={2025},
    eprint={2507.15365},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2507.15365},
}
```