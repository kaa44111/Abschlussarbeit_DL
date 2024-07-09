import os
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import numpy as np
from torchvision import transforms 

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, count=None):
        self.root_dir=root_dir
        self.transform = transform
        self.count = count

        # Pfade zu den Bildern und Masken
        self.image_folder = os.path.join(root_dir, 'grabs')
        self.mask_folder = os.path.join(root_dir, 'masks')

        # self.image_files = sorted(os.listdir(self.image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
        # self.mask_files = sorted(os.listdir(self.mask_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))

        # Liste aller Bilddateien
        all_image_files = sorted(os.listdir(self.image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))

        # Filtere die Bilddateien, für die auch Masken existieren
        self.image_files = []
        self.mask_files = []

        for image_file in all_image_files:
            base_name = os.path.splitext(image_file)[0]
            mask_name = f"{base_name}.tif"
            if os.path.exists(os.path.join(self.mask_folder, mask_name)):
                self.image_files.append(image_file)
                self.mask_files.append(mask_name)

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
            image = Image.open(img_name).convert('RGB')

            # Laden der Maske für dieses Bild
            mask_name = os.path.join(self.mask_folder, self.mask_files[idx])
            mask = Image.open(mask_name).convert('L')
            mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0

            if self.transform:
                image = self.transform(image)
                #mask = self.transform(mask)

            return image, mask
        
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None, None  # Return dummy values
        
#dataset = CustomDataset(root_dir='prepare/test',transform=transforms.ToTensor())
