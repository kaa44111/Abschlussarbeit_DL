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

from utils.data_utils import find_image_and_mask_files_folder
#from utils.data_utils import show_normalized_images ,show_image_and_mask, compute_mean_std, compute_mean_std_from_dataset

def downsample_image(input_path, scale_factor):
    '''
    Bilder werden kleiner skaliert
    '''
    # Bild laden
    img = Image.open(input_path)

    # Neue Größe berechnen
    new_size = (int(img.width / scale_factor), int(img.height / scale_factor))

    # Bildgröße ändern (runterskalieren)
    return img.resize(new_size, Image.Resampling.LANCZOS)

    # # Verkleinertes Bild speichern
    # downsampled_img.save(output_path)

def find_masks_to_image(root_dir, scale_factor):
    '''
    Findet die passende Masken, die gleich wie die Bilder heißen, und führt ein Binning aus. 
    Die verkleinerten Bilder & Masken werden in einem neuen Ordner gespeichert.
    '''
    image_folder, mask_folder, image_files, mask_files = find_image_and_mask_files_folder(root_dir)

    print(f"Found {len(image_files)} images")
    print(f"Found {len(mask_files)} masks")

    # Ordnername aus image_folder extrahieren
    folder_name = os.path.basename(root_dir.rstrip('/\\'))
    image_modified = f"data/data_modified/{folder_name}/test/image"
    mask_modified = f"data/data_modified/{folder_name}/test/mask"

    # Sicherstellen, dass die Ausgabeordner existieren
    os.makedirs(image_modified, exist_ok=True)
    os.makedirs(mask_modified, exist_ok=True)

    for idx in range(len(image_files)):
        img_name = os.path.join(image_folder, image_files[idx])
        output_path = f"{image_modified}/{image_files[idx]}"
        downsampled_img = downsample_image(img_name, scale_factor)
        downsampled_img.save(output_path)

        mask_name = os.path.join(mask_folder, mask_files[idx])
        output_path = f"{mask_modified}/{mask_files[idx]}"
        downsampled_mask = downsample_image(mask_name, scale_factor)
        downsampled_mask.save(output_path)


#_____________________________________________________
####### Try binning and saving Images####

# if __name__ == '__main__':
#     try: 
#         # Load images
#         scale_factor = 4  # Verkleinerungsfaktor (z.B. auf 1/4 der ursprünglichen Größe)
#         root_dir = 'data/WireCheck'

#         find_masks_to_image(root_dir,scale_factor)

#     except Exception as e:
#         print(f"An error occurred: {e}")

#_____________________________________________________
####### Try binning and saving Images####

# if __name__ == '__main__':
#     try: 
#         # Load images
#         scale_factor = 4  # Verkleinerungsfaktor (z.B. auf 1/4 der ursprünglichen Größe)
#         root_dir = 'data/WireCheck'
#         dataset_name = 'WireCheck'

#         train_dir=downsample_image_and_mask(root_dir,dataset_name, scale_factor)

#     except Exception as e:
#         print(f"An error occurred: {e}")
