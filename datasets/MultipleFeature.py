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
import numpy as np
from torch.utils.data import random_split
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

class Equalize(torch.nn.Module):
    def forward(self, img):
        return F.equalize(img)

class CustomDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, mask_transform=None, count=None, num_features=6):
        self.root_dir=root_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.count = count
        self.num_features = num_features

        # Pfade zu den Bildern und Masken
        self.image_folder = os.path.join(root_dir, 'geometry_shapes', 'train', 'grabs')
        self.mask_folder = os.path.join(root_dir, 'geometry_shapes', 'train', 'masks')

        self.image_files = sorted(os.listdir(self.image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
        self.mask_files = sorted(os.listdir(self.mask_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))

        # Begrenzen der Anzahl der Dateien, wenn count nicht None ist
        if self.count is not None:
            self.image_files = self.image_files[:self.count]
            self.mask_files = self.mask_files[:self.count * self.num_features]  # Annahme: 6 Masken pro Bild

        print(f"Found {len(self.image_files)} images")
        print(f"Found {len(self.mask_files)} masks")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            # Load the image
            img_name = os.path.join(self.image_folder, self.image_files[idx])
            image = Image.open(img_name).convert('L')
            image=tv_tensors.Image(image)
            
            #Load Masks
            masks = []
            base_name = self.image_files[idx].split('.')[0]
            a = base_name
            for i in range(0,self.num_features):
                # Laden der Maske für dieses Bild
                mask_name = os.path.join(self.mask_folder, f"{base_name}{i}.png")
                mask = Image.open(mask_name).convert('L')
                mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).float()

                # Debugging-Informationen zur Maske
                # print(f"Loaded mask {i} with shape: {mask_tensor.shape} and unique values: {torch.unique(mask_tensor)}")

                masks.append(mask_tensor)

            if self.image_transform:
                image = self.image_transform(image)

            if self.mask_transform:
                masks = [self.mask_transform(mask) for mask in masks]

            masks_tensor = torch.stack(masks, dim=0)  # Erzeugt einen Tensor der Form [6, 1, H, W]
            masks_tensor = masks_tensor.squeeze(1)  # Ändert die Form zu [6, H, W]

            return image, masks_tensor
        
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None,None  # Return dummy values



def get_data_loaders():

    trans_image = v2.Compose([
            #Equalize(),
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),
            #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    mask_trans = v2.Compose([
        v2.ToPureTensor(),
    ])


    train_set = CustomDataset('data', image_transform=trans_image, mask_transform=mask_trans,count=250)
    val_set = CustomDataset('data', image_transform=trans_image, mask_transform=mask_trans,count=60)
    

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    batch_size = 25

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    return dataloaders, image_datasets

def compute_mean_std(dataset):
    dataloader = DataLoader(dataset, batch_size=25, shuffle=False, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images_count = 0

    for images, _ in dataloader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean([2]).sum(0)
        std += images.std([2]).sum(0)
        total_images_count += images.size(0)

    mean /= total_images_count
    std /= total_images_count

    return mean, std

if __name__ == '__main__':
    
    try:
        dataloader, dataset=get_data_loaders()

        batch1=dataset['train']
        mean, std = compute_mean_std(batch1)
        print(mean)
        print(std)

        # Beispiel für den direkten Zugriff auf das erste Batch
        batch = next(iter(dataloader['train']))
       
        images,masks = batch
        print(images.shape)
        print(masks.shape)
        print(images[0][0].shape)

        first_image = images[20]  # Das erste Bild im Batch
        first_batch_masks = masks[20]  # Die Masken des ersten Bildes im Batch
        first_mask = masks[20][0]

        print(f"First image min: {images[20].min()}, max: {images[20].max()}")
        print(f"First mask min: {masks[20][0].min()}, max: {masks[20][0].max()}")

        num_masks = first_batch_masks.shape[0]  # Anzahl der Masken
        fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))

        # Das Bild anzeigen
        image_array = first_image.permute(1, 2, 0).cpu().detach().numpy()
        axes[0].imshow(image_array, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('Image')
        
        # Die Masken anzeigen
        for i in range(num_masks):
            mask = first_batch_masks[i]
            mask_array = mask.cpu().detach().numpy()
            axes[i + 1].imshow(mask_array, cmap='gray')
            axes[i + 1].axis('off')
            axes[i + 1].set_title(f'Mask {i+1}')
        
        plt.show()
            
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
        
# def get_dataloaders():

#     transformations = v2.Compose([
#             v2.ToPureTensor(),
#             v2.ToDtype(torch.float32, scale=True),
#             v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ])

#     custom_dataset = CustomDataset('data', transform=transformations)

#     # Definieren Sie die Größen für das Training und die Validierung
#     dataset_size = len(custom_dataset)
#     train_size = int(0.8 * dataset_size)
#     val_size = dataset_size - train_size

#     # Aufteilen des Datensatzes in Trainings- und Validierungsdaten
#     train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

#     # Ausgabe der Anzahl der Bilder in Trainings- und Validierungsdatensätzen
#     print(f"Anzahl der Bilder im Trainingsdatensatz: {len(train_dataset)}")
#     print(f"Anzahl der Bilder im Validierungsdatensatz: {len(val_dataset)}")

#     # Erstellen der DataLoader für Training und Validierung
#     train_loader = DataLoader(train_dataset, batch_size=25, shuffle=False)
#     val_loader = DataLoader(val_dataset, batch_size=25, shuffle=False)

#     #Creating Dataloaders:
#     dataloaders = {
#         'train': train_loader,
#         'val': val_loader
#     }

#     return dataloaders,custom_dataset
