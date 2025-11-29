"""
Download and prepare LLaVA datasets for Phase 1 and Phase 2 training.

Phase 1: LLaVA-Pretrain dataset (vision-language alignment)
Phase 2: LLaVA-Instruct-Mix dataset (instruction following)

Example usage:
    # Phase 1
    python scripts/prepare_dataset.py --phase 1 --dataset-dir ~/dataset/llava-pretrain

    # Phase 2
    python scripts/prepare_dataset.py --phase 2 --dataset-dir ~/dataset/llava-instruct-mix
"""

import os
import sys
import zipfile
import time
import argparse
import subprocess
import shutil
import json
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
        print("❌ Error: 'hf' command not found.")
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
                    remaining = (
                        (total_files - (i + 1)) / rate if rate > 0 else 0
                    )

                    progress_msg = (
                        f"\rProgress: {percent:.1f}% ({i + 1}/{total_files}) "
                        f"- {rate:.0f} files/s - ETA: {remaining:.0f}s"
                    )
                    sys.stdout.write(progress_msg)
                    sys.stdout.flush()

            print(f"\n✅ Extraction complete. Extracted to {extract_to}")

    except zipfile.BadZipFile:
        print("❌ Error: Bad zip file.")
    except Exception as e:
        print(f"❌ An error occurred during extraction: {e}")


def verify_phase1_dataset(dataset_dir):
    """Verifies that the Phase 1 dataset can be loaded correctly."""
    print(f"🔍 Verifying Phase 1 dataset in {dataset_dir}...")

    data_path = dataset_dir / "blip_laion_cc_sbu_558k.json"
    image_folder = dataset_dir

    if not data_path.exists():
        print(f"❌ Error: Data file {data_path} not found.")
        return False

    # Check if at least one image subdirectory exists (e.g. 00000)
    # Images are stored in numbered subdirectories like 00000, 00001, etc.
    if not (image_folder / "00000").exists():
        msg = f"❌ Error: Image subdirectories not found: {image_folder}/00000"
        print(msg)
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


def verify_phase2_dataset(dataset_dir):
    """Verifies that the Phase 2 dataset can be loaded correctly."""
    print(f"🔍 Verifying Phase 2 dataset in {dataset_dir}...")

    # Check if parquet files exist
    parquet_files = list(dataset_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"❌ Error: No parquet files found in {dataset_dir}")
        return False

    print(f"Found {len(parquet_files)} parquet file(s)")

    try:
        # Try to import datasets library
        try:
            from datasets import load_dataset
        except ImportError:
            print("❌ Error: 'datasets' library not found.")
            print("   Install with: pip install datasets")
            return False

        # Load dataset from local directory
        dataset = load_dataset(
            "parquet",
            data_files=str(parquet_files[0]),
            split="train"
        )

        print(f"✅ Dataset loaded successfully. Size: {len(dataset)}")
        print(f"Dataset features: {list(dataset.features.keys())}")

        # Verify a few samples
        num_samples = min(5, len(dataset))
        indices = random.sample(range(len(dataset)), num_samples)

        print(f"Testing {num_samples} samples...")

        for idx in indices:
            sample = dataset[idx]
            # Check that conversations and image exist
            # theblackcat102/llava-instruct-mix uses
            # 'conversations' and 'image'
            if 'conversations' not in sample:
                msg = (
                    f"❌ Error: 'conversations' field not found "
                    f"in sample {idx}"
                )
                print(msg)
                return False
            if 'image' not in sample:
                msg = (
                    f"❌ Error: 'image' field not found in sample {idx}"
                )
                print(msg)
                return False

            # Verify conversations is a list or string (JSON string)
            conversations = sample['conversations']
            if isinstance(conversations, str):
                # If it's a JSON string, try to parse it
                try:
                    conversations = json.loads(conversations)
                except json.JSONDecodeError:
                    msg = (
                        f"❌ Error: 'conversations' is not valid JSON "
                        f"in sample {idx}"
                    )
                    print(msg)
                    return False

            if not isinstance(conversations, list):
                msg = (
                    f"❌ Error: 'conversations' is not a list "
                    f"in sample {idx}"
                )
                print(msg)
                return False

            # Verify image exists (can be PIL Image, bytes, or path)
            image = sample['image']
            has_image = image is not None

            print(f"  Sample {idx}: {len(conversations)} conversation turns, "
                  f"has_image={has_image}")

        print("✅ Verification complete: Random samples loaded successfully.")
        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def prepare_phase1_dataset(
    dataset_dir, skip_download, skip_unzip, skip_verify
):
    """Prepare Phase 1 dataset (LLaVA-Pretrain)."""
    print("=" * 60)
    print("Phase 1: LLaVA-Pretrain Dataset Preparation")
    print("=" * 60)

    repo_id = "liuhaotian/LLaVA-Pretrain"
    zip_file = dataset_dir / "images.zip"

    if not skip_download:
        if zip_file.exists():
            print(f"✅ Dataset already downloaded at {zip_file}")
        elif not download_dataset(repo_id, str(dataset_dir)):
            sys.exit(1)

    if not skip_unzip:
        # Check if the dataset appears to be already extracted
        extracted_data_path = dataset_dir / "blip_laion_cc_sbu_558k.json"
        # Check for one of the image subdirectories
        extracted_image_dir_check = dataset_dir / "00000"

        if (extracted_data_path.exists() and
                extracted_image_dir_check.exists()):
            print(f"✅ Dataset already extracted to {dataset_dir}")
        elif zip_file.exists():
            print("Dataset not extracted. Extracting...")
            unzip_with_progress(str(zip_file), str(dataset_dir))
        else:
            print(f"⚠️  Zip file not found at {zip_file}. "
                  "Skipping extraction.")

    if not skip_verify:
        verify_phase1_dataset(dataset_dir)


def prepare_phase2_dataset(dataset_dir, skip_download, skip_verify):
    """Prepare Phase 2 dataset (LLaVA-Instruct-Mix)."""
    print("=" * 60)
    print("Phase 2: LLaVA-Instruct-Mix Dataset Preparation")
    print("=" * 60)

    repo_id = "theblackcat102/llava-instruct-mix"

    if not skip_download:
        # Check if dataset already exists
        parquet_files = list(dataset_dir.glob("*.parquet"))
        if parquet_files:
            print(f"✅ Dataset already downloaded at {dataset_dir}")
            print(f"   Found {len(parquet_files)} parquet file(s)")
        else:
            print(f"📦 Downloading {repo_id} to {dataset_dir}...")

            # Use hf CLI to preserve original parquet partition structure
            # This maintains the original file names like
            # train-00000-of-00093-*.parquet
            if not download_dataset(repo_id, str(dataset_dir)):
                print("❌ Download failed with hf CLI.")
                print("   Make sure 'hf' command is available:")
                print("   pip install huggingface-cli")
                print("   huggingface-cli login")

                # Fallback: try datasets library with sharding
                try:
                    from datasets import load_dataset
                    print("⚠️  Trying 'datasets' library as fallback...")
                    print(
                        "   Note: This will create new partition files, "
                        "not preserve original names"
                    )

                    os.makedirs(dataset_dir, exist_ok=True)

                    # Download dataset
                    dataset = load_dataset(repo_id, split="train")

                    # Save as parquet with sharding to preserve partitioned
                    # format. Use max_shard_size to create multiple files
                    dataset.to_parquet(
                        str(dataset_dir),
                        max_shard_size="500MB"
                    )

                    print(
                        f"✅ Dataset downloaded and saved to {dataset_dir}"
                    )
                    print("   Created partitioned parquet files")

                except ImportError:
                    print("❌ 'datasets' library not found.")
                    print("   Install with: pip install datasets")
                    sys.exit(1)
                except Exception as e:
                    print(f"❌ Download failed: {e}")
                    sys.exit(1)

    if not skip_verify:
        verify_phase2_dataset(dataset_dir)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare LLaVA datasets for Phase 1 and Phase 2."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1 (LLaVA-Pretrain)
  python scripts/prepare_dataset.py --phase 1 \\
      --dataset-dir /workspace/dataset/llava-pretrain

  # Phase 2 (LLaVA-Instruct-Mix-VSFT)
  python scripts/prepare_dataset.py --phase 2 \\
      --dataset-dir /workspace/dataset/llava-instruct-mix
        """
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        required=True,
        help="Training phase: 1 for pretrain, 2 for instruction following"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step."
    )
    parser.add_argument(
        "--skip-unzip",
        action="store_true",
        help="Skip the unzip step (Phase 1 only)."
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the verification step."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help=(
            "Root directory for the dataset. "
            "Defaults to ~/dataset/llava-pretrain (phase 1) or "
            "~/dataset/llava-instruct-mix (phase 2) if not specified."
        )
    )
    args = parser.parse_args()

    # Set default dataset directory based on phase
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        if args.phase == 1:
            dataset_dir = Path.home() / "dataset" / "llava-pretrain"
        else:
            dataset_dir = Path.home() / "dataset" / "llava-instruct-mix"

    # Ensure directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    if args.phase == 1:
        prepare_phase1_dataset(
            dataset_dir,
            args.skip_download,
            args.skip_unzip,
            args.skip_verify
        )
    elif args.phase == 2:
        prepare_phase2_dataset(
            dataset_dir,
            args.skip_download,
            args.skip_verify
        )


if __name__ == "__main__":
    main()
