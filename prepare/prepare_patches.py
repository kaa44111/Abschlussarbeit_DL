import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

from PIL import Image
import numpy as np
from utils.data_utils import find_image_and_mask_files_folder


# def create_patches(image, patch_size):
#     '''
#     Teilt ein Bild in Patches auf und füllt Ränder mit Nullen auf, um die gleiche Größe zu gewährleisten.
#     '''
#     width, height = image.size
#     patches = []
#     for i in range(0, width, patch_size):
#         for j in range(0, height, patch_size):
#             box = (i, j, min(i + patch_size, width), min(j + patch_size, height))
#             patch = image.crop(box)
            
#             # Create a new patch with the desired size and paste the cropped patch
#             padded_patch = Image.new('RGB' if image.mode == 'RGB' else 'L', (patch_size, patch_size))
#             padded_patch.paste(patch, (0, 0))
#             patches.append(padded_patch)
    
#     return patches

def create_patches(image, patch_size):
    '''
    Teilt ein Bild in Patches auf. Randbehandlung : Der Rand wird einfach weggelassen
    '''
    if patch_size is None:
        patch_size= 200
        print("Default if patch_size:200")

    width, height = image.size
    patches = []
    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            box = (i, j, i + patch_size, j + patch_size)
            if box[2] <= width and box[3] <= height:
                patch = image.crop(box)
                patches.append(patch)
    return patches

def save_patches(patches, base_name, output_folder, prefix):
    '''
    Speichert die Patches.
    '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for idx, patch in enumerate(patches):
        patch_name = f"{base_name}_{prefix}{idx + 1}.tif"
        patch.save(os.path.join(output_folder, patch_name))

def prepare_patches(root_dir, patch_size = None, dataset_name=None):
    '''
    Teilt die Bilder und Masken in Patches auf und speichert sie.
    '''
    image_folder, mask_folder, image_files, mask_files = find_image_and_mask_files_folder(root_dir, dataset_name)

    if dataset_name is None:
        dataset_name = os.path.basename(root_dir.rstrip('/\\'))

    # Ordnername aus image_folder extrahieren
    folder_name = f"data/data_modified/{dataset_name}/patched"    
    image_modified = f"{folder_name}/grabs"
    mask_modified = f"{folder_name}/masks"

    # Überprüfen, ob der Ordner bereits existiert
    if os.path.exists(folder_name):
        print(f"Der Ordner {folder_name} existiert bereits. Überspringen des Downsamplings.")
        return folder_name

    output_image_folder = os.path.join(image_modified)
    output_mask_folder = os.path.join(mask_modified)

    for image_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_file)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image_patches = create_patches(image, patch_size)
        mask_patches = create_patches(mask, patch_size)

        save_patches(image_patches, os.path.splitext(image_file)[0], output_image_folder, 'patch')
        save_patches(mask_patches, os.path.splitext(mask_file)[0], output_mask_folder, 'patch')

    return folder_name

def process_test_images(root_dir, patch_size):
    '''
    Teilt die Test Bilder in Patches auf und speichert sie.
    '''
    image_folder = os.path.join(root_dir, 'test')

    # Erstellen des neuen Verzeichnisses basierend auf root_dir
    base_name = os.path.basename(root_dir)
    output_image_folder = os.path.join('data/data_modified', base_name, 'test')

    for image_name in os.listdir(image_folder):
        if not image_name.endswith('.tif'):
            continue
        
        base_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(image_folder, image_name)
        
        image = Image.open(image_path).convert('RGB')
        
        image_patches = create_patches(image, patch_size)
        
        save_patches(image_patches, base_name, output_image_folder, 'patch')

# Beispielhafte Verwendung
root_dir = 'data/RetinaVessel'
patch_size = 256  # Größe der Patches

#process_images_and_masks(root_dir, patch_size)
#process_test_images(root_dir, patch_size)