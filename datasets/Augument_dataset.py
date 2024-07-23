import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from collections import defaultdict
import matplotlib.pyplot as plt

from train.train import run
from test_models.test_model import test
import numpy as np

import torch

import torch.utils
from PIL import Image
import numpy as np
from utils.data_utils import show_image_and_mask

class CustomDataset(Dataset):
    def __init__(self, root_dir, dataset_name=None, transform=None, count=None):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
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

        for image_file in all_image_files:
            base_name = os.path.splitext(image_file)[0]

            # Bestimme den Masken-Namen basierend auf dem dataset_name
            if dataset_name == "RetinaVessel":
                mask_name = f"{base_name}.tiff"
            elif dataset_name == "Ölflecken":
                mask_name = f"{base_name}_1.bmp"
            elif dataset_name == "circle_data":
                mask_name = f"{base_name}1.png"
            else:
                mask_name = f"{base_name}.tif"  # Gleicher Name wie das Bild

            if os.path.exists(os.path.join(self.mask_folder, mask_name)):
                #for _ in range(augment_times):
                self.image_files.append(image_file)
                self.mask_files.append(mask_name)
            else:
                raise FileNotFoundError(f"Mask file not found: {mask_name}")

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

            # Laden der Maske für dieses Bild
            mask_name = os.path.join(self.mask_folder, self.mask_files[idx])
            mask = Image.open(mask_name).convert('L')
            mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() #/ 255.0

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)

            return image, mask

        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None, None  # Return dummy values

'''!!To Do move specific Transformations to __main__ in the train.py or test.py'''
def get_dataloaders(root_dir,dataset_name=None, batch_size=25, transformations=None, split_size = None):
    '''
    Default Transform: ToPureTensor(), ToDtype(torch.float32, scale=True)
    Default batch_size : 15
    Default split_size : 0.8
    '''
    if transformations is None:
        transformations = v2.Compose([
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    
    trans1 = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomEqualize(p=1.0),
        v2.ToPureTensor(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    trans2 = v2.Compose([
        v2.RandomPhotometricDistort(p=1),
        v2.RandomHorizontalFlip(p=1),
        v2.ToPureTensor(),
        v2.ToDtype(torch.float32, scale=True),])

    dataset_original = CustomDataset(root_dir=root_dir, dataset_name=dataset_name, transform=transformations)
    dataset_flip = CustomDataset(root_dir=root_dir, dataset_name=dataset_name, transform=trans1)
    dataset_distort = CustomDataset(root_dir=root_dir, dataset_name=dataset_name, transform=trans2)

    custom_dataset = ConcatDataset([dataset_original,dataset_flip,dataset_distort])

    if split_size is None:
        split_size=0.8
    
    # Definieren Sie die Größen für das Training und die Validierung
    dataset_size = len(custom_dataset)
    train_size = int(split_size * dataset_size)

    # Generieren Sie reproduzierbare Indizes für Training und Validierung
    indices = list(range(dataset_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Erstellen Sie Subsets für Training und Validierung
    train_dataset = Subset(custom_dataset, train_indices)
    val_dataset = Subset(custom_dataset, val_indices)

    # Ausgabe der Anzahl der Bilder in Trainings- und Validierungsdatensätzen
    print(f"Anzahl der Bilder im Trainingsdatensatz: {len(train_dataset)}")
    print(f"Anzahl der Bilder im Validierungsdatensatz: {len(val_dataset)}")

    # Erstellen der DataLoader für Training und Validierung
    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=25, shuffle=True, num_workers=0)

    # Creating Dataloaders:
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    return dataloaders, custom_dataset

if __name__ == '__main__':
     try:
        from models.UNetBatchNorm import UNetBatchNorm
    
        train_dir = 'data_modified/Dichtflächen/patched'
        dataset_name = 'Dichtflächen'
    
        dataloader,custom_dataset = get_dataloaders(root_dir=train_dir,dataset_name=dataset_name)

        batch = next(iter(dataloader['train']))
        images,masks = batch
        print(images.shape)
        print(masks.shape)

        print(f"First image min: {images[0].min()}, max: {images[0].max()}")
        print(f"First mask min: {masks[0].min()}, max: {masks[0].max()}") 

        # Überprüfung der Bilder im Dataloader
        # for i in range(10,15):
        #     show_image_and_mask(images[i],masks[i])  

    #     #_____________________________________________________________
        save_name = 'test_UNetBatchNorm'

        ####Training für ein Modell Starten
        print("Train Model with Dichtflächen Dataset:")
        run(UNetBatchNorm, dataloader, dataset_name,save_name)

        trans = v2.Compose([
            #v2.RandomEqualize(p=1.0),
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        
    #     results_dir = os.path.join('train/results',dataset_name)
        trained_model = 'train/results/Dichtflächen/test_UNetBatchNorm.pth'
        
        
        test(UNet=UNetBatchNorm,test_dir=train_dir,transformations = trans,test_trained_model=trained_model)

    #     # #______________________________________________________________

    #     # #####Training für alle Modelle Starten
    #     # run_compare(dataloader,dataset_name)

    #     # #####Test für alle trainerte Modelle
    #     # test_compare(test_dir, dataset_name, UNet, UNetMaxPool, UNetBatchNorm)
        
     except Exception as e:
        print(f"An error occurred: {e}")
