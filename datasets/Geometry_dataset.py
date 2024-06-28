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
import re
from utils.data_utils import MAPPING, BinningTransform, PatchTransform
from torch.utils.data import random_split


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, mapping = None, count=None):
        self.root_dir=root_dir
        self.transform = transform
        self.count = count

        # Pfade zu den Bildern und Masken
        self.image_folder = os.path.join(root_dir, 'geometry_shapes', 'grabs')
        self.mask_folder = os.path.join(root_dir, 'geometry_shapes', 'masks')

        self.image_files = sorted(os.listdir(self.image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
        self.mask_files = sorted(os.listdir(self.mask_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))

        # Define the label mapping
        self.mapping = mapping

        # Begrenzen der Anzahl der Dateien, wenn count nicht None ist
        if self.count is not None:
            self.image_files = self.image_files[:self.count]
            self.mask_files = self.mask_files[:self.count * 6]  # Annahme: 6 Masken pro Bild

        print(f"Found {len(self.image_files)} images")
        print(f"Found {len(self.mask_files)} masks")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            # Load the image
            img_name = os.path.join(self.image_folder, self.image_files[idx])
            image = Image.open(img_name).convert('RGB')
            image=tv_tensors.Image(image)
            
            mask_paths = self.mask_files[idx * 6 : (idx + 1) * 6]
            
            masks = [Image.open(os.path.join(self.mask_folder, mask_file)).convert('L') for mask_file in mask_paths]
            masks = [tv_tensors.Mask(torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0) for mask in masks]

            # # Create a combined mask where each form gets a unique class
            # combined_mask = torch.zeros_like(masks[0], dtype=torch.long)  # Ensure combined_mask is long type
            # for i, mask in enumerate(masks):
            #     if self.mapping:
            #         combined_mask[mask > 0] = self.mapping.get(f"form_{i}", i+1)  # Use external mapping if available
            #     else:
            #         combined_mask[mask > 0] = i+1  # Use index if no mapping is provided

            if self.transform:
                image = self.transform(image)

            masks_tensor = torch.stack(masks, dim=0)  # Erzeugt einen Tensor der Form [6, 1, H, W]
            masks_tensor = masks_tensor.squeeze(1)  # Ändert die Form zu [6, H, W]

            return image, masks_tensor
        
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None,None  # Return dummy values
        

def get_dataloaders():

    transformations = v2.Compose([
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    custom_dataset = CustomDataset('data', transform=transformations,mapping=MAPPING)

    # Definieren Sie die Größen für das Training und die Validierung
    dataset_size = len(custom_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    # Aufteilen des Datensatzes in Trainings- und Validierungsdaten
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

    # Ausgabe der Anzahl der Bilder in Trainings- und Validierungsdatensätzen
    print(f"Anzahl der Bilder im Trainingsdatensatz: {len(train_dataset)}")
    print(f"Anzahl der Bilder im Validierungsdatensatz: {len(val_dataset)}")

    # Erstellen der DataLoader für Training und Validierung
    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=25, shuffle=False)

    #Creating Dataloaders:
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    return dataloaders

if __name__ == '__main__':
    

    try:
        transformations = v2.Compose([
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        dataset = CustomDataset('data', transform=transformations,count=4)
        sample = dataset[0]
        img, mas = sample
        print(mas.shape)  

        dataloader=get_dataloaders()
        # Beispiel für den direkten Zugriff auf das erste Batch
        batch = next(iter(dataloader['train']))
        images,masks = batch
        print(images.shape)
        print(masks.shape)


        #   # Optional: visualize the first sample in the batch
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.title("First Image")
        # plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())  # Converting to HWC for visualization

        # plt.subplot(1, 2, 2)
        # plt.title("First Combined Mask")
        # plt.imshow(combined_masks[0, 0].cpu().numpy(), cmap='gray')  # Displaying combined mask

        # plt.show()
        

    except Exception as e:
        print(f"An error occurred: {e}")