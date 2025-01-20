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
    model_dir = project_root / 'app' / 'models' / 'weights'
    zip_path = model_dir / 'model.zip'
    
    # Create model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Model download URL
    url = "https://download.wetransfer.com/eugv/8908841bef9fb09340466809dd3661af20250119201918/bc15990087893f1da88d5a8a5384c40099310a7b/exp1.zip?cf=y&token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImRlZmF1bHQifQ.eyJleHAiOjE3MzczNzg5NzIsImlhdCI6MTczNzM3ODM3MiwiZG93bmxvYWRfaWQiOiIyZTU4ZDUzYi0zYzM2LTQ1ZTMtOWM2Yi0zYTM4ZWNkZjYzZDgiLCJzdG9yYWdlX3NlcnZpY2UiOiJzdG9ybSJ9.aR5xvj-4hIpg8EmgZFYIkp2dckTRddYu52V9FVwhKbQ"

    print("Downloading model file...")
    if download_file(url, zip_path):
        print("Download completed. Extracting files...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            os.remove(zip_path)
            print("Model extraction completed successfully!")
        except zipfile.BadZipFile:
            print("Error: The downloaded file is not a valid zip file")
            os.remove(zip_path)
            sys.exit(1)
    else:
        print("Failed to download the model file")
        sys.exit(1)

if __name__ == "__main__":
    main()
