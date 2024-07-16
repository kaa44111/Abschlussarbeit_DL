import os
from PIL import Image
import numpy as np


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
        patch_name = f"{base_name}_{prefix}{idx + 1}.tiff"
        patch.save(os.path.join(output_folder, patch_name))

def process_images_and_masks(root_dir, patch_size):
    '''
    Teilt die Bilder und Masken in Patches auf und speichert sie.
    '''
    image_folder = os.path.join(root_dir, 'train', 'grabs')
    if not os.path.exists(image_folder):
        image_folder = os.path.join(root_dir, 'grabs')

    mask_folder = os.path.join(root_dir, 'train', 'masks')
    if not os.path.exists(mask_folder):
        mask_folder = os.path.join(root_dir,'masks')

    # Erstellen des neuen Verzeichnisses basierend auf root_dir
    base_name = os.path.basename(root_dir)
    output_image_folder = os.path.join('data_modified', base_name, 'train', 'grabs')
    output_mask_folder = os.path.join('data_modified', base_name, 'train', 'masks')

    for image_name in os.listdir(image_folder):
        if not image_name.endswith('.tif'):
            continue
        
        base_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(image_folder, image_name)
        mask_name = f"{base_name}_1.bmp"
        mask_path = os.path.join(mask_folder, mask_name)
        
        if not os.path.exists(mask_path):
            print(f"Maske für Bild {image_name} nicht gefunden. Überspringen.")
            continue
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        image_patches = create_patches(image, patch_size)
        mask_patches = create_patches(mask, patch_size)
        
        save_patches(image_patches, base_name, output_image_folder, 'patch')
        save_patches(mask_patches, base_name, output_mask_folder, 'patch')

def process_test_images(root_dir, patch_size):
    '''
    Teilt die Test Bilder in Patches auf und speichert sie.
    '''
    image_folder = os.path.join(root_dir, 'test')

    # Erstellen des neuen Verzeichnisses basierend auf root_dir
    base_name = os.path.basename(root_dir)
    output_image_folder = os.path.join('data_modified', base_name, 'test')

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