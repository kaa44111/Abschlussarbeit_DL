import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
    
import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms import v2
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import tv_tensors
from torch.utils.data import Dataset, DataLoader
from datasets.WireCheck_dataset import CustomDataset

def downsample_image(input_path, output_path, scale_factor):
    # Bild laden
    img = Image.open(input_path)
    
    # Neue Größe berechnen
    new_size = (int(img.width / scale_factor), int(img.height / scale_factor))
    
    # Bildgröße ändern (runterskalieren)
    downsampled_img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Verkleinertes Bild speichern
    downsampled_img.save(output_path)

def find_masks_to_image(image_folder,mask_folder,scale_factor):
    # Liste aller Bilddateien
    all_image_files = sorted(os.listdir(image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))

        # Filtere die Bilddateien, für die auch Masken existieren
    image_files = []
    mask_files = []

    for image_file in all_image_files:
        base_name = os.path.splitext(image_file)[0]
        mask_name = f"{base_name}.tif"
        if os.path.exists(os.path.join(mask_folder, mask_name)):
            image_files.append(image_file)
            mask_files.append(mask_name)

    print(f"Found {len(image_files)} images")
    print(f"Found {len(mask_files)} masks")
    image_modified = 'prepare/test/image'
    mask_modified = 'prepare/test/mask'
    images = []
    for idx in range(len(image_files)):
        img_name = os.path.join(image_folder,image_files[idx])
        images.append(img_name)
        downsample_image(img_name,f"{image_modified}/{image_files[idx]}",scale_factor)
        mask_name = os.path.join(mask_folder, mask_files[idx])
        downsample_image(mask_name,f"{mask_modified}/{mask_files[idx]}",scale_factor)


def compute_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False, num_workers=0)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0

    for images, _ in dataloader:
        # Reshape to (batch_size * height * width, channels)
        images = images.permute(0, 2, 3, 1).reshape(-1, 3)
        
        # Update total number of pixels
        total_pixels += images.shape[0]

        # Sum up mean and std for each channel
        mean += images.mean(dim=0)
        std += images.std(dim=0)

    # Calculate the mean and std across the entire dataset
    mean /= len(dataloader)
    std /= len(dataloader)

    return mean, std

def show_image_with_rgb(image_path):
    # Bild laden und in RGB umwandeln
    image = Image.open(image_path).convert('RGB')
    
    # Bild in ein NumPy-Array umwandeln
    np_image = np.array(image)
    
    # Anzeigen des Bildes
    plt.imshow(np_image)
    plt.title("Image")
    plt.axis('off')
    plt.show()
    
    # Anzeigen der RGB-Werte
    print("RGB values of the 1st Image:")
    print(np_image)

def show_image_and_mask(image, mask):
    # Rücktransformieren des Bildes (um die Normalisierung rückgängig zu machen)
    image = image.numpy().transpose((1, 2, 0))
    #image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)

    # Maske umwandeln
    mask = mask.squeeze().numpy()

    # Anzeigen des Bildes und der Maske
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    try: 
        # Load images
        scale_factor = 4  # Verkleinerungsfaktor (z.B. auf 1/4 der ursprünglichen Größe)
        i_path = 'data/WireCheck/grabs'
        m_path = 'data/WireCheck/masks'

        #find_masks_to_image(i_path,m_path,scale_factor)

        dataset = CustomDataset(root_dir='prepare/test_binning',transform=transforms.ToTensor())
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        mean, std = compute_mean_std(dataset)
        print(mean)
        print(std)

        # Beispielhafte Verwendung
        image_path = 'prepare/test_binning/grabs/01Grab.tiff'
        show_image_with_rgb(image_path)

        trans= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),
        ])

        dataset1=CustomDataset(root_dir='prepare/test_binning',transform=trans)
        image, mask = dataset1[0]
        print(image.shape)

        show_image_and_mask(image,mask)

       

#______________________________________
        # #Bild laden und in RGB umwandeln
        # img = Image.open(image_path).convert('RGB')

        # img_tensor=tv_tensors.Image(img)

        # # Gleichung auf das PIL-Bild anwenden
        # equalized_img = F.equalize(img_tensor)

        # # # Debugging-Ausgabe: minimale und maximale Werte des Tensors
        # print(f"Unequalized Image min: {img_tensor.min()}, max: {img_tensor.max()}")
        # print(f"Equalized Image min: {equalized_img.min()}, max: {equalized_img.max()}")

        # # Optional: zurück in ein PIL-Bild konvertieren und anzeigen
        # equalized_img = transforms.ToPILImage()(img_tensor)

        # # Zeige das Originalbild und das equalized Bild
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.title("Original Image")
        # plt.imshow(img)

        # plt.subplot(1, 2, 2)
        # plt.title("Equalized Image")
        # plt.imshow(equalized_img)
        # plt.show()
#_________________________________________________
        

    except Exception as e:
        print(f"An error occurred: {e}")

