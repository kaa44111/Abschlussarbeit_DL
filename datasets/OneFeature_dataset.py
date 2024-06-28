import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
    
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from utils.data_utils import BinningTransform, PatchTransform


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, count=None):
        self.root_dir=root_dir
        self.transform = transform
        self.count = count

        # Pfade zu den Bildern und Masken
        self.image_folder = os.path.join(root_dir, 'grabs')
        self.mask_folder = os.path.join(root_dir, 'masks')

        self.image_files = sorted(os.listdir(self.image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
        self.mask_files = sorted(os.listdir(self.mask_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))

        # Begrenzen der Anzahl der Dateien, wenn count nicht None ist
        if self.count is not None:
            self.image_files = self.image_files[:self.count]
            self.mask_files = self.mask_files[:self.count] 

        print(f"Found {len(self.image_files)} images")
        print(f"Found {len(self.mask_files)} masks")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            # Laden des Bildes
            img_name = os.path.join(self.image_folder, self.image_files[idx])
            image = Image.open(img_name)
            image = tv_tensors.Image(image)

            # Laden der Maske für dieses Bild
            base_name = self.image_files[idx].split('.')[0]
            mask_name = os.path.join(self.mask_folder, f"{base_name}1.png")
            mask = Image.open(mask_name)
            mask = tv_tensors.Mask(torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0)

            if self.transform:
                image = self.transform(image)
                #mask = self.transform(mask)

            return image, mask
        
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None, None  # Return dummy values
        
def get_dataloaders():
    # use the same transformations for train/val in this example
    transformations = v2.Compose([
        v2.ToPureTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = CustomDataset('data/circle_data/train', transform = transformations)
    val_set = CustomDataset('data/circle_data/val', transform = transformations)

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    batch_size = 25

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    return dataloaders

if __name__ == '__main__':
    try:
        transformations = v2.Compose([
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        
        dataset = CustomDataset(root_dir='data/circle_data/train',transform=transformations)
        image, masks_tensor = dataset[0]
        dataloader = get_dataloaders()
        batch = next(iter(dataloader['train']))
        images, masks =batch 
        if image is not None and masks_tensor is not None:
            print("Datasets:")
            print(image.shape)
            print(masks_tensor.shape)

        if images is not None and masks is not None:
            print("Dataloaders:")
            print(images.shape)
            print(masks.shape)
        else:
            print("Failed to load data")
        
    except Exception as e:
        print(f"An error occurred: {e}")


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
