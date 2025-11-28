"""
Download and prepare LLaVA-Pretrain dataset.

Example usage:
    python scripts/prepare_dataset.py --dataset-dir /workspace/dataset/llava-pretrain
"""

import os
import sys
import zipfile
import time
import argparse
import subprocess
import shutil
from pathlib import Path
import random

# Add src to path so we can import vlm
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from vlm.data import LLaVAPretrainDataset

def download_dataset(repo_id, local_dir, repo_type="dataset"):
    """
    Downloads the dataset using hf download command.
    
    Args:
        repo_id: Repository ID (e.g., "liuhaotian/LLaVA-Pretrain")
        local_dir: Local directory to save the dataset
        repo_type: Type of repository ("dataset" or "model")
    
    Returns:
        bool: True if download succeeded, False otherwise
    """
    print(f"📦 Downloading {repo_id} to {local_dir}...")
    
    # Check if hf command is available
    hf_cmd = shutil.which("hf")
    if hf_cmd is None:
        print("❌ Error: 'hf' command not found. Please install huggingface-cli.")
        print("   Install with: pip install huggingface-cli")
        return False
    
    try:
        # Ensure local_dir exists
        os.makedirs(local_dir, exist_ok=True)
        
        # Build hf download command
        cmd = [
            hf_cmd,
            "download",
            repo_id,
            "--repo-type", repo_type,
            "--local-dir", local_dir,
            "--resume-download"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Download complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ hf download failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def unzip_with_progress(zip_path, extract_to):
    """Unzips a file with a progress bar."""
    if not os.path.exists(zip_path):
        print(f"Error: File {zip_path} not found.")
        return

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    print(f"📂 Opening {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            file_list = zf.infolist()
            total_files = len(file_list)
            print(f"Total files to extract: {total_files}")
            
            start_time = time.time()
            for i, file_info in enumerate(file_list):
                zf.extract(file_info, extract_to)
                
                # Update progress every 100 files or if it's the last one
                if (i + 1) % 100 == 0 or (i + 1) == total_files:
                    percent = (i + 1) / total_files * 100
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    remaining = (total_files - (i + 1)) / rate if rate > 0 else 0
                    
                    sys.stdout.write(f"\rProgress: {percent:.1f}% ({i + 1}/{total_files}) - {rate:.0f} files/s - ETA: {remaining:.0f}s")
                    sys.stdout.flush()
            
            print(f"\n✅ Extraction complete. Extracted to {extract_to}")

    except zipfile.BadZipFile:
        print("❌ Error: Bad zip file.")
    except Exception as e:
        print(f"❌ An error occurred during extraction: {e}")

def verify_dataset(dataset_dir):
    """Verifies that the dataset can be loaded correctly."""
    print(f"🔍 Verifying dataset in {dataset_dir}...")
    
    data_path = dataset_dir / "blip_laion_cc_sbu_558k.json"
    image_folder = dataset_dir
    
    if not data_path.exists():
        print(f"❌ Error: Data file {data_path} not found.")
        return False
        
    # Check if at least one image subdirectory exists (e.g. 00000)
    # The images are stored in numbered subdirectories like 00000, 00001, etc.
    if not (image_folder / "00000").exists():
        print(f"❌ Error: Image subdirectories (e.g., {image_folder}/00000) not found.")
        return False
        
    try:
        dataset = LLaVAPretrainDataset(
            data_path=str(data_path),
            image_folder=str(image_folder)
        )
        
        print(f"✅ Dataset loaded successfully. Size: {len(dataset)}")
        
        # Verify a few samples
        indices = list(range(min(5, len(dataset))))
        if len(dataset) > 10:
            indices.extend(random.sample(range(5, len(dataset)), 5))
            
        print(f"Testing {len(indices)} samples...")
        
        for idx in indices:
            sample = dataset[idx]
            # Just access the data to make sure it loads
            _ = sample['raw_image']
            _ = sample['raw_text']
            
        print("✅ Verification complete: Random samples loaded successfully.")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download and prepare LLaVA-Pretrain dataset.")
    parser.add_argument("--skip-download", action="store_true", help="Skip the download step.")
    parser.add_argument("--skip-unzip", action="store_true", help="Skip the unzip step.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip the verification step.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help=(
            "Root directory for the dataset. "
            "Defaults to ~/dataset/llava-pretrain if not specified."
        )
    )
    args = parser.parse_args()

    repo_id = "liuhaotian/LLaVA-Pretrain"
    # Use custom dataset directory if provided, otherwise use ~/dataset/llava-pretrain
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        dataset_dir = Path.home() / "dataset" / "llava-pretrain"
    zip_file = dataset_dir / "images.zip"

    if not args.skip_download:
        if zip_file.exists():
            print(f"✅ Dataset already downloaded at {zip_file}")
        elif not download_dataset(repo_id, str(dataset_dir)):
            sys.exit(1)

    if not args.skip_unzip:
        # Check if the dataset appears to be already extracted
        extracted_data_path = dataset_dir / "blip_laion_cc_sbu_558k.json"
        extracted_image_dir_check = dataset_dir / "00000" # Check for one of the image subdirectories

        if extracted_data_path.exists() and extracted_image_dir_check.exists():
            print(f"✅ Dataset already extracted to {dataset_dir}")
        elif zip_file.exists():
            print(f"Dataset not extracted. Extracting...")
            unzip_with_progress(str(zip_file), str(dataset_dir))
        else:
            print(f"⚠️  Zip file not found at {zip_file}. Skipping extraction.")

    if not args.skip_verify:
        verify_dataset(dataset_dir)

if __name__ == "__main__":
    main()
