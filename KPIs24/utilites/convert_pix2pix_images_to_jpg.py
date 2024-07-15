import os
from PIL import Image

def convert_png_to_jpg(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            png_path = os.path.join(directory, filename)
            jpg_path = os.path.join(directory, filename.replace(".png", ".jpg"))
            try:
                with Image.open(png_path) as img:
                    if 'mask' in png_path:
                        img = img.convert("L")  # Convert to grayscale
                    else:
                        img = img.convert("RGB")  # Convert to RGB
                    img.save(jpg_path, "JPEG")
                os.remove(png_path)  # Remove the original PNG file
                print(f"Converted {filename} to JPG format.")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")
                
def main():
    datadir = "/mnt/atlas/data_KPIs/data/KPIs24_Training_Data/Task1_patch_level/pix2pix_data_augmentation_rearranged/"
    for root, dirs, files in os.walk(datadir):
        for file in files:
            if file.endswith(".png"):
                convert_png_to_jpg(root)
                break
            
if __name__ == "__main__":
    main()