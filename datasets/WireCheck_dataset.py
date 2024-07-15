import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
    
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import v2
from torchvision import tv_tensors
from torch.utils.data import random_split
from utils.data_utils import show_image_and_mask, compute_mean_std
import torchvision.transforms.functional as F
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, count=None):
        self.root_dir=root_dir
        self.transform = transform
        self.count = count

        # Pfade zu den Bildern und Masken
        self.image_folder = os.path.join(root_dir, 'grabs')
        self.mask_folder = os.path.join(root_dir, 'masks')

        # Liste aller Bilddateien
        all_image_files = sorted(os.listdir(self.image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))

        # Filtere die Bilddateien, für die auch Masken existieren
        self.image_files = []
        self.mask_files = []
        i=1
        for image_file in all_image_files:
            base_name = os.path.splitext(image_file)[0]
            mask_name = f"{base_name}.tiff" #for RetinaVessel
            #mask_name = f"{base_name}.tif" #for Wirecheck
            #mask_name = f'{base_name}_1.bmp' #for Ölflecken
            #mask_name = f"{base_name}1.png" #for circle_data
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
            image = v2.functional.pil_to_tensor(image).float()/ 255.0
            #image= tv_tensors.Image(image)

            # Laden der Maske für dieses Bild
            mask_name = os.path.join(self.mask_folder, self.mask_files[idx])
            mask = Image.open(mask_name).convert('L')
            mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0

            if self.transform:
                image = self.transform(image)
            # else:
            #     image = F.to_tensor(image)

            return image, mask
        
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None, None  # Return dummy values
        

def get_dataloaders(root_dir):
    #mean, std = compute_mean_std(os.path.join(root_dir, 'grabs'))

    transformations = v2.Compose([
        v2.RandomEqualize(p=1.0),
        v2.ToPureTensor(),
        v2.ToDtype(torch.float32, scale=True),
        #v2.Normalize(mean=mean, std=std)
    ])

    # transformations = transforms.Compose([
        
    #     transforms.ToTensor(),
    #     v2.RandomEqualize(p=1.0),
    # ])

    custom_dataset = CustomDataset(root_dir=root_dir, transform=transformations)

    # Definieren Sie die Größen für das Training und die Validierung
    dataset_size = len(custom_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    # Generieren Sie reproduzierbare Indizes für Training und Validierung
    indices = list(range(dataset_size))
    # random.seed(42)  # Sicherstellen, dass die Aufteilung jedes Mal gleich ist
    # random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Erstellen Sie Subsets für Training und Validierung
    train_dataset = Subset(custom_dataset, train_indices)
    val_dataset = Subset(custom_dataset, val_indices)

    # Ausgabe der Anzahl der Bilder in Trainings- und Validierungsdatensätzen
    print(f"Anzahl der Bilder im Trainingsdatensatz: {len(train_dataset)}")
    print(f"Anzahl der Bilder im Validierungsdatensatz: {len(val_dataset)}")

    # Erstellen der DataLoader für Training und Validierung
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=True, num_workers=0)

    # Creating Dataloaders:
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    return dataloaders, custom_dataset

        
if __name__ == '__main__':
    
    try:
        root_dir = 'data_modified/RetinaVessel/train'
        dataloader,custom_dataset = get_dataloaders(root_dir=root_dir)

        image, mask =custom_dataset[0]
        print('Image:')
        print(image.shape)
        print(image.min(), image.max())
        print('Mask:')
        print(mask.shape)
        print(mask.min(), mask.max())

        # Überprüfung der Bilder im Dataset
        # for i in range(72,76):
        #     image,mask = custom_dataset[i]
        #     show_image_and_mask(image,mask) 

        batch = next(iter(dataloader['train']))
        images,masks = batch
        print(images.shape)
        print(masks.shape)

        print(f"First image min: {images[0].min()}, max: {images[0].max()}")
        print(f"First mask min: {masks[0].min()}, max: {masks[0].max()}") 

        # # Überprüfung der Bilder im Dataloader
        # for i in range(4):
        #     show_image_and_mask(images[i],masks[i])  

    except Exception as e:
        print(f"An error occurred: {e}")


### Get a Random split in the dataset
# def get_dataloaders(root_dir):

#     mean, std = compute_mean_std(os.path.join(root_dir, 'grabs'))

#     transformations = v2.Compose([
#             v2.ToPureTensor(),
#             v2.ToDtype(torch.float32, scale=True),
#             v2.Normalize(mean=mean, std=std),
#             ])

#     custom_dataset = CustomDataset(root_dir=root_dir, transform=transformations)

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
