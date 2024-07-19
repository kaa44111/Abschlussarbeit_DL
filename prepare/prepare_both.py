import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

from utils.data_utils import find_image_and_mask_files_folder
# from prepare.prepare_binning import prepare_binning
# from prepare.prepare_patches import prepare_patches,save_patches
    
from PIL import Image
import numpy as np

def extract_patches(image, patch_size, use_padding=False):
    width, height = image.size
    if use_padding:
        patches = []
        for i in range(0, width, patch_size):
            for j in range(0, height, patch_size):
                box = (i, j, min(i + patch_size, width), min(j + patch_size, height))
                patch = image.crop(box)
                
                # Create a new patch with the desired size and paste the cropped patch
                padded_patch = Image.new('RGB' if image.mode == 'RGB' else 'L', (patch_size, patch_size))
                padded_patch.paste(patch, (0, 0))
                patches.append(padded_patch)
    else:
        patches = []
        for i in range(0, width, patch_size):
            for j in range(0, height, patch_size):
                box = (i, j, i + patch_size, j + patch_size)
                if box[2] <= width and box[3] <= height:
                    patch = image.crop(box)
                    patches.append(patch)

    return patches

def downsample_image(img,scale_factor):
    # Calculate new size
    new_size = (int(img.width / scale_factor), int(img.height / scale_factor))
        
    # Resize the image
    return img.resize(new_size, Image.Resampling.LANCZOS)

def process_images(root_dir, dataset_name, downsample_factor=None, patch_size=None, use_padding=False):
    image_folder, mask_folder, image_files, mask_files = find_image_and_mask_files_folder(root_dir, dataset_name)
    
    output_base = f"data_modified/{dataset_name}"
    if downsample_factor and patch_size:
        output_dir = f"{output_base}/processed"
    elif downsample_factor:
        output_dir = f"{output_base}/downsampled"
    elif patch_size:
        output_dir = f"{output_base}/patched"
    else:
        raise ValueError("Entweder downsample_factor oder patch_size muss angegeben werden.")

    if os.path.exists(output_dir):
        print(f"Verzeichnis {output_dir} existiert bereits. Keine weiteren Operationen werden durchgeführt.")
        return output_dir

    #os.makedirs(output_dir, exist_ok=True)

    image_modified = f"{output_dir}/grabs"
    mask_modified = f"{output_dir}/masks"

    # Sicherstellen, dass die Ausgabeordner existieren
    os.makedirs(image_modified, exist_ok=True)
    os.makedirs(mask_modified, exist_ok=True)

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if downsample_factor is not None:
            img = downsample_image(img,downsample_factor)
            mask = downsample_image(mask,downsample_factor)

        if patch_size:
            img_patches = extract_patches(img, patch_size)
            mask_patches = extract_patches(mask, patch_size)

            for i, (img_patch, mask_patch) in enumerate(zip(img_patches, mask_patches)):
                img_name = f"{os.path.splitext(img_file)[0]}_patch{i+1}.tif"
                mask_name = f"{os.path.splitext(mask_file)[0]}_patch{i+1}.tif"
                
                output_img_path = os.path.join(image_modified,img_name)
                output_mask_path = os.path.join(mask_modified,mask_name)
                img_patch.save(output_img_path)
                mask_patch.save(output_mask_path)
        else:
            output_img_path = os.path.join(image_modified, img_file)
            output_mask_path = os.path.join(mask_modified, mask_file)
            
            img.save(output_img_path)
            mask.save(output_mask_path)

    print(f"Verarbeitung abgeschlossen. Ergebnisse gespeichert in: {output_dir}")
    return output_dir

# def prepare_both(root_dir, scale_factor=None, patch_size=None, dataset_name=None):
#     '''
#     Führt das Downsampling durch, teilt die Bilder und Masken in Patches auf und speichert sie.
#     '''
#     image_folder, mask_folder, image_files, mask_files = find_image_and_mask_files_folder(root_dir,dataset_name)

#     if dataset_name is None:
#         dataset_name = os.path.basename(root_dir.rstrip('/\\'))

#     # Ordnername aus image_folder extrahieren
#     folder_name = f"data_modified/{dataset_name}/downsampled_patched"
#     image_modified = f"{folder_name}/grabs"
#     mask_modified = f"{folder_name}/masks"

#     # Überprüfen, ob der Ordner bereits existiert
#     if os.path.exists(folder_name):
#         print(f"Der Ordner {folder_name} existiert bereits. Überspringen des Downsamplings.")
#         return folder_name

#     # Sicherstellen, dass die Ausgabeordner existieren
#     os.makedirs(image_modified, exist_ok=True)
#     os.makedirs(mask_modified, exist_ok=True)

#     print(f"Found {len(image_files)} images")
#     print(f"Found {len(mask_files)} masks")

#     for idx in range(len(image_files)):
#         img_name = os.path.join(image_folder, image_files[idx])
#         downsampled_img_path = f"{image_modified}/{image_files[idx]}"
#         mask_name = os.path.join(mask_folder, mask_files[idx])
#         downsampled_mask_path = f"{mask_modified}/{mask_files[idx]}"

#         # Downsampling für Bild und Maske durchführen
#         downsample_image(img_name, downsampled_img_path, scale_factor)
#         downsample_image(mask_name, downsampled_mask_path, scale_factor)

#         # Patches erstellen und speichern
#         img = Image.open(downsampled_img_path).convert('RGB')
#         mask = Image.open(downsampled_mask_path).convert('L')

#         image_patches = create_patches(img, patch_size)
#         mask_patches = create_patches(mask, patch_size)

#         save_patches(image_patches, os.path.splitext(image_files[idx])[0], image_modified, 'patch')
#         save_patches(mask_patches, os.path.splitext(mask_files[idx])[0], mask_modified, 'patch')

#     return folder_name