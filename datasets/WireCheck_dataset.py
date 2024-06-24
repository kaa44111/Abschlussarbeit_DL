import os
import sys
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
from torch.utils.data import random_split
from torchvision.transforms import ToTensor


# Load images
image_path = 'data/WireCheck/grabs/00Grab (2).tiff'
mask_path = 'data/WireCheck/masks/00Grab (2).tif'

try:
    image = Image.open(image_path)
    mask = Image.open(mask_path)
except Exception as e:
    print(f"Error loading images: {e}")

# Convert images to tensors
image_tensor = tv_tensors.Image(image)
mask_tensor = tv_tensors.Mask(mask)

# Check tensor shapes
print(f"Image tensor shape: {image_tensor.shape}")
print(f"Mask tensor shape: {mask_tensor.shape}")

# Create heatmap image in red channel
heatmap = torch.empty_like(image_tensor)
heatmap[0, :, :] = mask_tensor[0, :, :]  # Red channel
heatmap[:, :, 0] = 0  # Green channel
heatmap[:, 0, :] = 0  # Blue channel

print(f"Heatmaptensor shape: {heatmap.shape}")


import torchvision.transforms.functional as TF
img = TF.to_pil_image(image_tensor)  # assuming your image in x
h_img = TF.to_pil_image(heatmap)

res = Image.blend(img, h_img, 0.5)

# Plotting
plt.figure(figsize=(16, 8))

# Plot image
plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(image_tensor.permute(1, 2, 0).cpu().numpy())

# Plot mask
plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(mask_tensor.squeeze().cpu().numpy(), cmap='gray')

# Plot mask
plt.subplot(1, 3, 3)
plt.title("Heatmap")
plt.imshow(res , cmap='gray')

plt.show()

# class CustomDataset(Dataset):
#     def __init__(self, root_dir, transform=None, mapping = None):
#         self.root_dir=root_dir
#         self.transform = transform

#         # Pfade zu den Bildern und Masken
#         self.image_files = list(sorted(os.listdir(os.path.join(root_dir, "grabs"))))
#         self.mask_file = list(sorted(os.listdir(os.path.join(root_dir, "masks"))))

#         # Define the label mapping
#         self.mapping = mapping

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         try:
#             # Load the image
#             img_name = os.path.join(self.image_folder, self.image_files[idx])
#             image = Image.open(img_name).convert('RGB')
#             image=tv_tensors.Image(image)
            
#             mask_paths = self.mask_files[idx * 6 : (idx + 1) * 6]
            
#             masks = [Image.open(os.path.join(self.mask_folder, mask_file)).convert('L') for mask_file in mask_paths]
#             masks = [tv_tensors.Mask(torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0) for mask in masks]

#             # Create a combined mask where each form gets a unique class
#             combined_mask = torch.zeros_like(masks[0], dtype=torch.long)  # Ensure combined_mask is long type
#             for i, mask in enumerate(masks):
#                 if self.mapping:
#                     combined_mask[mask > 0] = self.mapping.get(f"form_{i}", i+1)  # Use external mapping if available
#                 else:
#                     combined_mask[mask > 0] = i+1  # Use index if no mapping is provided

#             if self.transform:
#                 image = self.transform(image)
#                 combined_mask = self.transform(combined_mask)
#                 masks = [self.transform(mask) for mask in masks]

#             masks_tensor = torch.stack(masks, dim=0)  # Erzeugt einen Tensor der Form [6, 1, H, W]
#             masks_tensor = masks_tensor.squeeze(1)  # Ändert die Form zu [6, H, W]

#             return image, masks_tensor, combined_mask
        
#         except Exception as e:
#             print(f"Error loading data at index {idx}: {e}")
#             return None, None, None  # Return dummy values
        

# def get_dataloaders():

#     transformations = v2.Compose([
#     v2.ToPureTensor(),
#     BinningTransform(bin_size=2),  # Beispiel für Binning mit bin_size 2
#     v2.ToDtype(torch.float32, scale=True),
#     #PatchTransform(patch_size=64),  # Beispiel für das Aufteilen in Patches der Größe 64x64
#     ])

#     custom_dataset = CustomDataset('data', transform=transformations,mapping=MAPPING)

#     # Definieren Sie die Größen für das Training und die Validierung
#     dataset_size = len(custom_dataset)
#     train_size = int(0.8 * dataset_size)
#     val_size = dataset_size - train_size

#     # Aufteilen des Datensatzes in Trainings- und Validierungsdaten
#     train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

#     # Erstellen der DataLoader für Training und Validierung
#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,collate_fn=custom_collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,collate_fn=custom_collate_fn)

#     #Creating Dataloaders:
#     dataloaders = {
#         'train': train_loader,
#         'val': val_loader
#     }

#     return dataloaders

# if __name__ == '__main__':
#     try:
#        dataset = CustomDataset(root_dir='data/WireCheck')

#     except Exception as e:
#         print(f"An error occurred: {e}")