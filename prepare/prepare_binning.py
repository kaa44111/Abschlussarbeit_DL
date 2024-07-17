import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
    
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision import transforms
from datasets.OneFeature import CustomDataset
from utils.data_utils import show_normalized_images ,show_image_and_mask, compute_mean_std, compute_mean_std_from_dataset

def downsample_image(input_path, output_path, scale_factor):
    '''
    Bilder werden kleiner skaliert
    '''
    # Bild laden
    img = Image.open(input_path)
    
    # Neue Größe berechnen
    new_size = (int(img.width / scale_factor), int(img.height / scale_factor))
    
    # Bildgröße ändern (runterskalieren)
    downsampled_img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Verkleinertes Bild speichern
    downsampled_img.save(output_path)

def find_masks_to_image(root_dir, scale_factor):
    '''
    Findet die passende Masken, die gleich wie die Bilder heißen, und führt ein Binning aus. 
    Die verkleinerten Bilder & Masken werden in einem neuen Ordner gespeichert.
    '''
    image_folder = os.path.join(root_dir, 'train', 'grabs')
    if not os.path.exists(image_folder):
        image_folder = os.path.join(root_dir, 'grabs')

    mask_folder = os.path.join(root_dir, 'train', 'masks')
    if not os.path.exists(mask_folder):
        mask_folder = os.path.join(root_dir,'masks')
        
    # Liste aller Bilddateien
    all_image_files = sorted(os.listdir(image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))

    # Filtere die Bilddateien, für die auch Masken existieren
    image_files = []
    mask_files = []

    for image_file in all_image_files:
        base_name = os.path.splitext(image_file)[0]
        mask_name = f"{base_name}.tif"
        if os.path.exists(os.path.join(mask_folder, mask_name)):
            image_files.append(image_file)
            mask_files.append(mask_name)

    print(f"Found {len(image_files)} images")
    print(f"Found {len(mask_files)} masks")

    # Ordnername aus image_folder extrahieren
    folder_name = os.path.basename(root_dir.rstrip('/\\'))
    image_modified = f"data_modified/{folder_name}/image"
    mask_modified = f"data_modified/{folder_name}/mask"

    # Sicherstellen, dass die Ausgabeordner existieren
    os.makedirs(image_modified, exist_ok=True)
    os.makedirs(mask_modified, exist_ok=True)

    for idx in range(len(image_files)):
        img_name = os.path.join(image_folder, image_files[idx])
        downsample_image(img_name, f"{image_modified}/{image_files[idx]}", scale_factor)
        mask_name = os.path.join(mask_folder, mask_files[idx])
        downsample_image(mask_name, f"{mask_modified}/{mask_files[idx]}", scale_factor)


#_____________________________________________________
####### Try binning and saving Images####

if __name__ == '__main__':
    try: 
        # Load images
        scale_factor = 4  # Verkleinerungsfaktor (z.B. auf 1/4 der ursprünglichen Größe)
        root_dir = 'data/WireCheck'

        find_masks_to_image(root_dir,scale_factor)

    except Exception as e:
        print(f"An error occurred: {e}")
