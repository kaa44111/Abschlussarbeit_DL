import os
from PIL import Image
import numpy as np

def create_patches(image, patch_size):
    '''
    Teilt ein Bild in Patches auf.
    '''
    width, height = image.size
    patches = []
    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            box = (i, j, min(i + patch_size, width), min(j + patch_size, height))
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

def process_images_and_masks(image_folder, mask_folder, output_image_folder, output_mask_folder, patch_size):
    '''
    Teilt die Bilder und Masken in Patches auf und speichert sie.
    '''
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, image_name)
        
        if not os.path.exists(mask_path):
            print(f"Maske für Bild {image_name} nicht gefunden. Überspringen.")
            continue
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        image_patches = create_patches(image, patch_size)
        mask_patches = create_patches(mask, patch_size)
        
        base_name = os.path.splitext(image_name)[0]
        
        save_patches(image_patches, base_name, output_image_folder, 'patch')
        save_patches(mask_patches, base_name, output_mask_folder, 'patch')

# Beispielhafte Verwendung
image_folder = 'prepare/test_binning/grabs'
mask_folder = 'prepare/test_binning/masks'
output_image_folder = 'path_to_your_output_folder/modified/grabs'
output_mask_folder = 'path_to_your_output_folder/modified/masks'
patch_size = 256  # Größe der Patches

process_images_and_masks(image_folder, mask_folder, output_image_folder, output_mask_folder, patch_size)