import os
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import numpy as np
from torchvision import transforms 
from tqdm import tqdm

def compute_mean_std(dataloader):
    '''
    Annahme dass die Bilder die gleiche height und width haben.
    '''
    # Initialize sums and squared sums for each channel
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    total_pixels = 0

    for batch_images, _ in tqdm(dataloader):  # (B,C,H,W)
        # Summe der Pixelwerte pro Kanal
        channels_sum += batch_images.sum(dim=[0, 2, 3])
        channels_squared_sum += (batch_images ** 2).sum(dim=[0, 2, 3])
        total_pixels += batch_images.numel() / batch_images.size(1)  # Anzahl der Pixel pro Kanal

    mean = channels_sum / total_pixels
    std = (channels_squared_sum / total_pixels - mean ** 2) ** 0.5

    return mean, std


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
        i=1
        for image_file in all_image_files:
            base_name = os.path.splitext(image_file)[0]
            mask_name = f"{base_name}.tif" #for Wirecheck
            #mask_name = f'{base_name}_1.bmp' #for Ölflecken
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
        

# transform = transforms.Compose([
#     transforms.Resize((192,192)),  # Alle Bilder auf dieselbe Größe bringen
#     transforms.ToTensor()
# ])

# dataset = CustomDataset(root_dir='data/Ölflecken',transform=transform)
# image, mask =dataset[0]
# print(image.shape)
# print(image.min(), image.max())
# print(mask.shape)
# print(mask.min(), mask.max())

# dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
# mean, std = compute_mean_std(dataloader)
# print(f"Mean: {mean}")
# print(f"Std: {std}")


# #__________________________________________
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(std=std, mean=mean)
# ])
# dataset1 = CustomDataset(root_dir='data/Ölflecken',transform=transform)

# import matplotlib.pyplot as plt

# def show_image(image):
#     image = image.numpy().transpose((1, 2, 0))
#     plt.imshow(image)
#     plt.show()

# # Beispielhafte Überprüfung der ersten Bilder im Dataset
# for i in range(4):
#     image, _ = dataset1[i]
#     show_image(image)