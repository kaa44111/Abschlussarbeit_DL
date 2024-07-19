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
from utils.data_utils import show_normalized_images ,show_image_and_mask, compute_mean_std, compute_mean_std_from_dataset

def downsample_image(input_path,scale_factor=2):
    '''
    Downsample an image using Pillow's LANCZOS resampling.
    '''
    if scale_factor is None:
        scale_factor=2

    with Image.open(input_path) as img:
        # Check if the image has already been downsampled
        if img.width < img.width / scale_factor or img.height < img.height / scale_factor:
            print(f"{input_path} has already been downsampled.")
            return img.copy()
        
        # Calculate new size
        new_size = (int(img.width / scale_factor), int(img.height / scale_factor))
        
        # Resize the image
        return img.resize(new_size, Image.Resampling.LANCZOS)

def save_downsample(downsampled_img,output_path):
    """
    Save the downsampled image, preserving the original format.
    """
    # Determine the file extension
    _, ext = os.path.splitext(output_path)

    if ext == '.bmp':
        downsampled_img.save(output_path, format='BMP')
    elif ext == '.tif' or ext == '.tiff':
        downsampled_img.save(output_path, format='TIFF', compression='tiff_deflate')
    else:
        # For other formats, use the default saving method
        downsampled_img.save(output_path)


def prepare_binning(root_dir, scale_factor=2, dataset_name=None):
    '''
    Perform downsampling on images and masks in the given directory.
    Default downsample : scale_factor = 2
    '''
    if dataset_name is None:
        dataset_name = os.path.basename(root_dir.rstrip('/\\'))    

    image_folder, mask_folder,image_files, mask_files=find_image_and_mask_files_folder(root_dir,dataset_name)

    # Ordnername aus image_folder extrahieren
    folder_name = f"data_modified/{dataset_name}/downsampled"    
    image_modified = f"{folder_name}/grabs"
    mask_modified = f"{folder_name}/masks"    

    # Überprüfen, ob der Ordner bereits existiert
    if os.path.exists(folder_name):
        print(f"Der Ordner {folder_name} existiert bereits. Überspringen des Downsamplings.")
        return folder_name
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")

    # Sicherstellen, dass die Ausgabeordner existieren
    os.makedirs(image_modified, exist_ok=True)
    os.makedirs(mask_modified, exist_ok=True)

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)
        output_img_path = os.path.join(image_modified, img_file)
        output_mask_path = os.path.join(mask_modified, mask_file)

        # Downsample and save image
        downsampled_image = downsample_image(img_path, scale_factor)
        save_downsample(downsampled_image, output_img_path)

        # Downsample and save mask
        downsampled_mask = downsample_image(mask_path, scale_factor)
        save_downsample(downsampled_mask, output_mask_path)

    return folder_name

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
