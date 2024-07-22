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
    patches = []
    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            box = (i, j, min(i + patch_size, width), min(j + patch_size, height))
            patch = image.crop(box)
            if use_padding and (box[2] - box[0] < patch_size or box[3] - box[1] < patch_size):
                padded_patch = Image.new(image.mode, (patch_size, patch_size))
                padded_patch.paste(patch, (0, 0))
                patches.append(padded_patch)
            else:
                patches.append(patch)
    return patches

def downsample_image(img, scale_factor):
    new_size = (int(img.width / scale_factor), int(img.height / scale_factor))
    return img.resize(new_size, Image.Resampling.LANCZOS)

def process_images(root_dir, dataset_name, downsample_factor=None, patch_size=None, use_padding=False):
    image_folder, mask_folder, image_files, mask_files = find_image_and_mask_files_folder(root_dir, dataset_name)
    
    # Debugging-Ausgaben hinzufügen
    print(f"Image Folder: {image_folder}")
    print(f"Mask Folder: {mask_folder}")
    print(f"Image Files: {image_files}")
    print(f"Mask Files: {mask_files}")
    
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
    
    image_modified = os.path.join(output_dir, "grabs")
    mask_modified = os.path.join(output_dir, "masks")

    os.makedirs(image_modified, exist_ok=True)
    os.makedirs(mask_modified, exist_ok=True)

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        # Überprüfen, ob die Bild- und Maskenpfade existieren
        if not os.path.exists(img_path):
            print(f"Bilddatei nicht gefunden: {img_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"Maskendatei nicht gefunden: {mask_path}")
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            # Debugging-Ausgaben
            print(f"Verarbeite Bild: {img_path}")
            print(f"Bildgröße vor Downsampling: {img.size}")
            print(f"Maskengröße vor Downsampling: {mask.size}")

            if downsample_factor is not None:
                img = downsample_image(img, downsample_factor)
                mask = downsample_image(mask, downsample_factor)

                # Debugging-Ausgaben
                print(f"Bildgröße nach Downsampling: {img.size}")
                print(f"Maskengröße nach Downsampling: {mask.size}")

            if patch_size:
                img_patches = extract_patches(img, patch_size, use_padding)
                mask_patches = extract_patches(mask, patch_size, use_padding)

                for i, (img_patch, mask_patch) in enumerate(zip(img_patches, mask_patches)):
                    img_name = f"{os.path.splitext(img_file)[0]}_patch{i+1}.tif"
                    mask_name = f"{os.path.splitext(mask_file)[0]}_patch{i+1}.tif"
                    
                    output_img_path = os.path.join(image_modified, img_name)
                    output_mask_path = os.path.join(mask_modified, mask_name)
                    
                    # Debugging-Ausgaben
                    print(f"Speichern des Bildes: {output_img_path}")
                    print(f"Speichern der Maske: {output_mask_path}")
                    
                    img_patch.save(output_img_path)
                    mask_patch.save(output_mask_path)
            else:
                output_img_path = os.path.join(image_modified, img_file)
                output_mask_path = os.path.join(mask_modified, mask_file)
                
                # Debugging-Ausgaben
                print(f"Speichern des Bildes: {output_img_path}")
                print(f"Speichern der Maske: {output_mask_path}")
                
                img.save(output_img_path)
                mask.save(output_mask_path)
        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei {img_file} oder {mask_file}: {e}")

    print(f"Verarbeitung abgeschlossen. Ergebnisse gespeichert in: {output_dir}")
    return output_dir

# Beispielaufruf der Funktion
# process_images("path_to_root_dir", "dataset_name", downsample_factor=2, patch_size=256, use_padding=True)

root_dir= 'data_modified/RetinaVessel'
dataset_name = 'RetinaVessel'

#Prepare Dataset (downsampe, batch, both)
'''
Default downsample : scale_factor = 2
Default patch:  patch_size= 200
'''
train_dir = process_images(root_dir,dataset_name,patch_size=256,use_padding=True)

#_______________________________________________________________
# def downsample_image(img, scale_factor, min_size=(32, 32)):
#     new_size = (max(int(img.width / scale_factor), min_size[0]), 
#                 max(int(img.height / scale_factor), min_size[1]))
#     return img.resize(new_size, Image.Resampling.LANCZOS)

# def process_images_for_binning(root_dir, dataset_name, downsample_factor):
#     # Verzeichnisse für Eingabebilder und Ausgabebilder
#     image_folder = os.path.join(root_dir, dataset_name, 'grabs')
#     mask_folder = os.path.join(root_dir, dataset_name, 'masks')
    
#     output_image_folder = os.path.join('data_modified', dataset_name, 'downsampled', 'grabs')
#     output_mask_folder = os.path.join('data_modified', dataset_name, 'downsampled', 'masks')
    
#     # Erstellen der Ausgabeordner, falls sie nicht existieren
#     os.makedirs(output_image_folder, exist_ok=True)
#     os.makedirs(output_mask_folder, exist_ok=True)
    
#     # Liste der Bilddateien im Verzeichnis
#     image_files = [f for f in os.listdir(image_folder) if f.endswith(('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
#     mask_files = [f for f in os.listdir(mask_folder) if f.endswith(('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    
#     for img_file, mask_file in zip(image_files, mask_files):
#         img_path = os.path.join(image_folder, img_file)
#         mask_path = os.path.join(mask_folder, mask_file)
        
#         # Überprüfen, ob die Bild- und Maskenpfade existieren
#         if not os.path.exists(img_path):
#             print(f"Bilddatei nicht gefunden: {img_path}")
#             continue
#         if not os.path.exists(mask_path):
#             print(f"Maskendatei nicht gefunden: {mask_path}")
#             continue

#         try:
#             img = Image.open(img_path)
#             mask = Image.open(mask_path)

#             # Debugging-Ausgaben
#             print(f"Verarbeite Bild: {img_path}")
#             print(f"Bildgröße vor Downsampling: {img.size}")
#             print(f"Maskengröße vor Downsampling: {mask.size}")

#             img = downsample_image(img, downsample_factor)
#             output_img_path = os.path.join(output_image_folder, img_file)
#              # Debugging-Ausgaben
#             print(f"Speichern des Bildes: {output_img_path}")
#             print(f"Bildgröße nach Downsampling: {img.size}")
#             img.save(output_img_path)

#             mask = downsample_image(mask, downsample_factor)
#             print(f"Maskengröße nach Downsampling: {mask.size}")
#             output_mask_path = os.path.join(output_mask_folder, mask_file)
#             print(f"Speichern der Maske: {output_mask_path}")
#             mask.save(output_mask_path)

#         except Exception as e:
#             print(f"Fehler beim Verarbeiten der Datei {img_file} oder {mask_file}: {e}")

#     print(f"Verarbeitung abgeschlossen. Ergebnisse gespeichert in: {os.path.join('data_modified', dataset_name, 'downsampled')}")
#     return os.path.join('data_modified', dataset_name, 'downsampled')

# process_images_for_binning("data", "Dichtflächen", downsample_factor=2)

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