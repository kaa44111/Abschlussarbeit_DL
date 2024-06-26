import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
    
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import re
from utils.data_utils import MAPPING, custom_collate_fn, BinningTransform, PatchTransform
import shutil
from sklearn.model_selection import train_test_split

# # Load images
# image_path = 'data/WireCheck/grabs/00Grab (2).tiff'
# mask_path = 'data/WireCheck/masks/00Grab (2).tif'

# try:
#     image = Image.open(image_path)
#     mask = Image.open(mask_path)
# except Exception as e:
#     print(f"Error loading images: {e}")

# # Convert images to tensors
# image_tensor = tv_tensors.Image(image)
# mask_tensor = tv_tensors.Mask(mask)

# # Check tensor shapes
# print(f"Image tensor shape: {image_tensor.shape}")
# print(f"Mask tensor shape: {mask_tensor.shape}")

# # Create heatmap image in red channel
# heatmap = torch.empty_like(image_tensor)
# heatmap[0, :, :] = mask_tensor[0, :, :]  # Red channel
# heatmap[:, :, 0] = 0  # Green channel
# heatmap[:, 0, :] = 0  # Blue channel

# print(f"Heatmaptensor shape: {heatmap.shape}")


# import torchvision.transforms.functional as TF
# img = TF.to_pil_image(image_tensor)  # assuming your image in x
# h_img = TF.to_pil_image(heatmap)

# res = Image.blend(img, h_img, 0.5)

# # Plotting
# plt.figure(figsize=(16, 8))

# # Plot image
# plt.subplot(1, 3, 1)
# plt.title("Image")
# plt.imshow(image_tensor.permute(1, 2, 0).cpu().numpy())

# # Plot mask
# plt.subplot(1, 3, 2)
# plt.title("Mask")
# plt.imshow(mask_tensor.squeeze().cpu().numpy(), cmap='gray')

# # Plot mask
# plt.subplot(1, 3, 3)
# plt.title("Heatmap")
# plt.imshow(res , cmap='gray')

# plt.show()


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Pfade zu den Bildern und Masken
        self.image_folder = os.path.join(root_dir, 'grabs')
        self.mask_folder = os.path.join(root_dir, 'mask_circle')

        # Sortieren der Dateien numerisch basierend auf den Ziffern im Dateinamen
        self.image_files = sorted(os.listdir(self.image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
        self.mask_files = sorted(os.listdir(self.mask_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))

        print(f"Found {len(self.image_files)} images")
        print(f"Found {len(self.mask_files)} masks")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            # Laden des Bildes
            img_name = os.path.join(self.image_folder, self.image_files[idx])
            image = Image.open(img_name).convert('RGB')
            image = tv_tensors.Image(image)

            # Laden der Maske für dieses Bild
            mask_name = os.path.join(self.mask_folder, f"{self.image_files[idx+1].split('.')[0]}1.png")
            mask = Image.open(mask_name).convert('L')
            mask = tv_tensors.Mask(torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0)
            
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)

            return image, mask
        
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None, None  # Return dummy values

if __name__ == '__main__':
    try:
        transformations = v2.Compose([
            v2.ToPureTensor(),
            BinningTransform(2),
            PatchTransform(30),
            v2.ToDtype(torch.float32, scale=True),
        ])
        
        dataset = CustomDataset(root_dir='data/geometry_shapes',transform=transformations)
        image, masks_tensor = dataset[0]
        if image is not None and masks_tensor is not None:
            print(image.shape)
            print(masks_tensor.shape)
        else:
            print("Failed to load data")
        
    except Exception as e:
        print(f"An error occurred: {e}")