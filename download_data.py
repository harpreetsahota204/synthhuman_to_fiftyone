"""Download specific chunks from the SynthHuman dataset.

Usage:
    python download_synthhuman_chunks.py ./data 0 20 40
    python download_synthhuman_chunks.py ./data sample
    
Chunk ranges:
    0-19:  Face samples
    20-39: Full body samples  
    40-59: Upper body samples
"""

import argparse
import subprocess
from pathlib import Path
from zipfile import ZipFile


def download_and_extract(chunk: str, data_dir: Path) -> None:
    """Download and extract a chunk."""
    filename = "SynthHuman_sample.zip" if chunk == "sample" else f"SynthHuman_{int(chunk):04d}.zip"
    url = f"https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/data/{filename}"
    zip_path = data_dir / filename
    
    print(f"Downloading {filename}...")
    subprocess.run([
        "wget", url, "-O", str(zip_path),
        "--no-check-certificate", "--continue", "--secure-protocol=TLSv1_2"
    ], check=True)
    
    print(f"Extracting {filename}...")
    with ZipFile(zip_path) as zf:
        zf.extractall(data_dir / "SynthHuman")
    
    zip_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Download SynthHuman chunks")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("chunks", nargs="+", help="Chunk numbers (0-59) or 'sample'")
    args = parser.parse_args()
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    for chunk in args.chunks:
        download_and_extract(chunk, args.output_dir)
    
    print("Done!")


if __name__ == "__main__":
    main()