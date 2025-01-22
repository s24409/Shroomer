import requests
import os
import sys
from pathlib import Path
import zipfile
from tqdm import tqdm

def download_file(url, destination):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get the total file size
        file_size = int(response.headers.get('content-length', 0))

        # Show the progress bar
        progress = tqdm(total=file_size, unit='iB', unit_scale=True)
        
        with open(destination, 'wb') as f:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                progress.update(size)
        progress.close()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

def main():
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / 'app' / 'models' / 'weights' / 'exp1'
    zip_path = model_dir / 'model.zip'
    
    # Create model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Model download URL
    url = "https://huggingface.co/s24409/fungi/resolve/main/model.pth?download=true"

    print("Downloading model file...")
    if download_file(url, model_dir / 'best_f1.pth'):
        print("Download completed")
    else:
         print("Failed to download the model file")
         sys.exit(1)

if __name__ == "__main__":
    main()
