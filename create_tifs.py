import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

print("Step 1: Starting script")
print("Current working directory:", os.getcwd())

folder = "C:/vscode/college/data1"
print(f"Step 2: Checking or creating folder: {folder}")

try:
    os.makedirs(folder, exist_ok=True)
    print("Step 3: Folder created or already exists")
except Exception as e:
    print("Step 3: Error creating folder:", e)
    exit(1)

new_path = f"{folder}/new_image.tif"
old_path = f"{folder}/old_mask.tif"

print("Step 4: Creating dummy data arrays")

width, height = 100, 100
transform = from_origin(0, 100, 1, 1)

new_data = np.random.randint(0, 255, (height, width), dtype=np.uint8)
old_data = new_data.copy()
old_data[20:40, 20:40] = 0

print("Step 5: Saving GeoTIFFs")

try:
    for path, data in [(new_path, new_data), (old_path, old_data)]:
        with rasterio.open(
            path, "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs="+proj=latlong",
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        print(f"Saved {path}")
except Exception as e:
    print("Error saving GeoTIFFs:", e)
    exit(1)

print("Step 6: Validation step")

def validate_tif(path):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return
    size_kb = os.path.getsize(path) / 1024
    with rasterio.open(path) as src:
        print(f"\nFile: {os.path.basename(path)}")
        print(f"  Dimensions: {src.width}x{src.height}")
        print(f"  File size: {size_kb:.2f} KB")
        print(f"  CRS: {src.crs}")
        print(f"  Dtype: {src.dtypes[0]}")


validate_tif(new_path)
validate_tif(old_path)

print("Step 7: Script finished successfully")
