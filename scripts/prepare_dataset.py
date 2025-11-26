import os
import sys
import zipfile
import time
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def download_dataset(repo_id, local_dir, repo_type="dataset"):
    """Downloads the dataset using huggingface_hub."""
    print(f"📦 Downloading {repo_id} to {local_dir}...")
    try:
        snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, resume_download=True)
        print("✅ Download complete.")
        return True
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

def main():
    parser = argparse.ArgumentParser(description="Download and prepare LLaVA-Pretrain dataset.")
    parser.add_argument("--skip-download", action="store_true", help="Skip the download step.")
    parser.add_argument("--skip-unzip", action="store_true", help="Skip the unzip step.")
    args = parser.parse_args()

    repo_id = "liuhaotian/LLaVA-Pretrain"
    # Determine project root relative to this script
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset" / "llava-pretrain"
    zip_file = dataset_dir / "images.zip"

    if not args.skip_download:
        if zip_file.exists():
            print(f"✅ Dataset already downloaded at {zip_file}")
        elif not download_dataset(repo_id, str(dataset_dir)):
            sys.exit(1)

    if not args.skip_unzip:
        if zip_file.exists():
            unzip_with_progress(str(zip_file), str(dataset_dir))
        else:
            print(f"⚠️  Zip file not found at {zip_file}. Skipping extraction.")

if __name__ == "__main__":
    main()
