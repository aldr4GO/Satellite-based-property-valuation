import requests
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_single_image(row, output_folder, mapbox_token):
    """Download a single image (to be run in parallel)"""
    house_id = row['id']
    lat = row['lat']
    lon = row['long']
    
    satellite_style = "satellite-v9"
    zoom = "16"
    pitch = "0"
    resolution = "1024"

    filename = f"{output_folder}/{house_id}.jpg"
    
    # Skip if exists
    if os.path.exists(filename):
        return {"status": "skipped", "house_id": house_id}

    url = f"https://api.mapbox.com/styles/v1/mapbox/{satellite_style}/static/{lon},{lat},{zoom},{pitch}/{resolution}x{resolution}?access_token={mapbox_token}"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return {"status": "success", "house_id": house_id}
        else:
            return {"status": "failed", "house_id": house_id, 
                    "code": response.status_code, "msg": response.text[:100]}
    except Exception as e:
        return {"status": "error", "house_id": house_id, "error": str(e)}


def download_images(excel_path, output_folder, mapbox_token, max_workers=6):
    """
    Download images in parallel using ThreadPoolExecutor
    
    Args:
        excel_path: Path to CSV file
        output_folder: Output directory
        mapbox_token: Mapbox API token
        max_workers: Number of parallel threads (default: 6)
    """
    # Load Data
    if not os.path.exists(excel_path):
        print(f"Error: File not found at {excel_path}")
        return

    df = pd.read_csv(excel_path)
    os.makedirs(output_folder, exist_ok=True)

    print(f"Starting parallel download for {len(df)} properties using {max_workers} threads...")
    
    # Statistics
    stats = {"success": 0, "failed": 0, "error": 0, "skipped": 0}
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(download_single_image, row, output_folder, mapbox_token): idx 
            for idx, row in df.iterrows()
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(df), desc="Downloading") as pbar:
            for future in as_completed(futures):
                result = future.result()
                stats[result["status"]] += 1
                
                # Print errors/failures
                if result["status"] == "failed":
                    print(f"\nFailed {result['house_id']}: {result['code']} → {result['msg']}")
                elif result["status"] == "error":
                    print(f"\nError {result['house_id']}: {result['error']}")
                
                pbar.update(1)
    
    print(f"\n{'='*50}")
    print(f"Download completed!")
    print(f"Success: {stats['success']} | Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']} | Errors: {stats['error']}")
    print(f"{'='*50}")

if __name__ == "__main__":
    output_folder = "images/train"
    mapbox_token = "<MAPBOX_TOKEN>"
    excel_path = "train.csv"
    
    # Use 6 workers to match your --cpus-per-task=6
    download_images(excel_path, output_folder, mapbox_token, max_workers=6)