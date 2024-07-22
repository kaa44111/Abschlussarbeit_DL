import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
    
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision import transforms
from datasets.OneFeature import CustomDataset
from utils.data_utils import show_normalized_images ,show_image_and_mask, compute_mean_std, compute_mean_std_from_dataset

# if __name__ == '__main__':
#     try: 
#         root_dir = 'data_modified/RetinaVessel/train'
#         image_folder = os.path.join(root_dir, 'grabs')  # Verzeichnis mit deinen Bildern

#         trans= transforms.Compose([
#             transforms.ToTensor(),
#             v2.RandomEqualize(p=1.0)
#         ])

#         dataset=CustomDataset(root_dir=root_dir,transform=trans)
#         mean, std = compute_mean_std_from_dataset(dataset)
#         print(f"Mean: {mean}")
#         print(f"Std: {std}")

#         trans= transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean,std=std),
#         ])

#         dataset1=CustomDataset(root_dir=root_dir,transform=trans)
#         image,mask = dataset1[0]
#         print(image.min(), image.max())

#         # for i in range(76,80):
#         #     unnormalized_image,_ = dataset[i]
#         #     normalized_image, mask = dataset1[i]
#         #     show_normalized_images(unnormalized_image, normalized_image, mask)
#         #     #print(mask.min(), mask.max())


#     except Exception as e:
#         print(f"An error occurred: {e}")

#________________________________________________________________
#########Test different Normailzation methods###########
# if __name__ == '__main__':
#     try:
#         root_dir = 'data_modified/RetinaVessel/train'
#         image_folder = os.path.join(root_dir, 'grabs')  # Verzeichnis mit deinen Bildern

#         trans= transforms.Compose([
#             transforms.ToTensor()
#         ])

#         dataset=CustomDataset(root_dir=root_dir,transform=trans)

#         mean1, std1 = compute_mean_std(image_folder)
#         mean, std = compute_mean_std_from_dataset(dataset)

#         normalization_methods = {
#             "Random Equalize": v2.RandomEqualize(p=1.0),  # Anwenden der Histogramm-Gleichverteilung
#             "Normalize from Dataset" : transforms.Normalize(mean=mean, std=std),
#             "Normalize from folder" : transforms.Normalize(mean=mean1, std=std1)
#         }

#         for method_name, method in normalization_methods.items():
#             print(f"Applying {method_name}...")
#             # trans = v2.Compose([
#             #     v2.ToPureTensor(),
#             #     v2.ToDtype(torch.float32, scale=True),
#             #     method,
#             # ])

#             trans = transforms.Compose([
#                 transforms.ToTensor(),
#                 method,
#             ])

#             dataset1 = CustomDataset(root_dir=root_dir, transform=trans)

#              # Anzeigen der Bilder und Masken
#             for i in range(76,80):
#                 unnormalized_image, _ = dataset[i]
#                 normalized_image, mask = dataset1[i]
#                 if unnormalized_image is None or normalized_image is None or mask is None:
#                     print(f"Error loading data at index {i}")
#                     continue
#                 print(f"Using method: {method_name}")
#                 show_normalized_images(unnormalized_image, normalized_image, mask)

#     except Exception as e:
#         print(f"An error occurred: {e}")


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
        

def downsample_image(input_path, output_path, scale_factor):
    '''
    Bilder werden kleiner skaliert
    '''
    # Bild laden
    img = Image.open(input_path)

    # Neue Größe berechnen
    new_size = (int(img.width / scale_factor), int(img.height / scale_factor))

    # Bildgröße ändern (runterskalieren)
    downsampled_img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Verkleinertes Bild speichern
    downsampled_img.save(output_path)

def find_masks_to_image(root_dir, scale_factor):
    '''
    Findet die passende Masken, die gleich wie die Bilder heißen, und führt ein Binning aus. 
    Die verkleinerten Bilder & Masken werden in einem neuen Ordner gespeichert.
    '''
    image_folder = os.path.join(root_dir, 'train', 'grabs')
    if not os.path.exists(image_folder):
        image_folder = os.path.join(root_dir, 'grabs')

    mask_folder = os.path.join(root_dir, 'train', 'masks')
    if not os.path.exists(mask_folder):
        mask_folder = os.path.join(root_dir,'masks')

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

    # Ordnername aus image_folder extrahieren
    folder_name = os.path.basename(root_dir.rstrip('/\\'))
    image_modified = f"data_modified/{folder_name}/test/image"
    mask_modified = f"data_modified/{folder_name}/test/mask"

    # Sicherstellen, dass die Ausgabeordner existieren
    os.makedirs(image_modified, exist_ok=True)
    os.makedirs(mask_modified, exist_ok=True)

    for idx in range(len(image_files)):
        img_name = os.path.join(image_folder, image_files[idx])
        downsample_image(img_name, f"{image_modified}/{image_files[idx]}", scale_factor)
        mask_name = os.path.join(mask_folder, mask_files[idx])
        downsample_image(mask_name, f"{mask_modified}/{mask_files[idx]}", scale_factor)

#_____________________________________________________
####### Try binning and saving Images####

if __name__ == '__main__':
    try: 
        # Load images
        scale_factor = 4  # Verkleinerungsfaktor (z.B. auf 1/4 der ursprünglichen Größe)
        root_dir = 'data/WireCheck'

        find_masks_to_image(root_dir,scale_factor)

    except Exception as e:
        print(f"An error occurred: {e}")
#__________________________________________________________________________________________________
# def create_patches(image, patch_size):
#     '''
#     Teilt ein Bild in Patches auf und füllt Ränder mit Nullen auf, um die gleiche Größe zu gewährleisten.
#     '''
#     width, height = image.size
#     patches = []
#     for i in range(0, width, patch_size):
#         for j in range(0, height, patch_size):
#             box = (i, j, min(i + patch_size, width), min(j + patch_size, height))
#             patch = image.crop(box)
            
#             # Create a new patch with the desired size and paste the cropped patch
#             padded_patch = Image.new('RGB' if image.mode == 'RGB' else 'L', (patch_size, patch_size))
#             padded_patch.paste(patch, (0, 0))
#             patches.append(padded_patch)
    
#     return patches

# def create_patches(image, patch_size):
#     '''
#     Teilt ein Bild in Patches auf. Randbehandlung : Der Rand wird einfach weggelassen
#     '''
#     width, height = image.size
#     patches = []
#     for i in range(0, width, patch_size):
#         for j in range(0, height, patch_size):
#             box = (i, j, i + patch_size, j + patch_size)
#             if box[2] <= width and box[3] <= height:
#                 patch = image.crop(box)
#                 patches.append(patch)
#     return patches

# def save_patches(patches, base_name, output_folder, prefix):
#     '''
#     Speichert die Patches.
#     '''
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     for idx, patch in enumerate(patches):
#         patch_name = f"{base_name}_{prefix}{idx + 1}.tiff"
#         patch.save(os.path.join(output_folder, patch_name))

# def process_images_and_masks(root_dir, patch_size):
#     '''
#     Teilt die Bilder und Masken in Patches auf und speichert sie.
#     '''
#     image_folder = os.path.join(root_dir, 'train', 'grabs')
#     if not os.path.exists(image_folder):
#         image_folder = os.path.join(root_dir, 'grabs')

#     mask_folder = os.path.join(root_dir, 'train', 'masks')
#     if not os.path.exists(mask_folder):
#         mask_folder = os.path.join(root_dir,'masks')

#     # Erstellen des neuen Verzeichnisses basierend auf root_dir
#     base_name = os.path.basename(root_dir)
#     output_image_folder = os.path.join('data_modified', base_name, 'train', 'grabs')
#     output_mask_folder = os.path.join('data_modified', base_name, 'train', 'masks')

#     for image_name in os.listdir(image_folder):
#         if not image_name.endswith('.tif'):
#             continue
        
#         base_name = os.path.splitext(image_name)[0]
#         image_path = os.path.join(image_folder, image_name)
#         mask_name = f"{base_name}_1.bmp"
#         mask_path = os.path.join(mask_folder, mask_name)
        
#         if not os.path.exists(mask_path):
#             print(f"Maske für Bild {image_name} nicht gefunden. Überspringen.")
#             continue
        
#         image = Image.open(image_path).convert('RGB')
#         mask = Image.open(mask_path).convert('L')
        
#         image_patches = create_patches(image, patch_size)
#         mask_patches = create_patches(mask, patch_size)
        
#         save_patches(image_patches, base_name, output_image_folder, 'patch')
#         save_patches(mask_patches, base_name, output_mask_folder, 'patch')

# def process_test_images(root_dir, patch_size):
#     '''
#     Teilt die Test Bilder in Patches auf und speichert sie.
#     '''
#     image_folder = os.path.join(root_dir, 'test')

#     # Erstellen des neuen Verzeichnisses basierend auf root_dir
#     base_name = os.path.basename(root_dir)
#     output_image_folder = os.path.join('data_modified', base_name, 'test')

#     for image_name in os.listdir(image_folder):
#         if not image_name.endswith('.tif'):
#             continue
        
#         base_name = os.path.splitext(image_name)[0]
#         image_path = os.path.join(image_folder, image_name)
        
#         image = Image.open(image_path).convert('RGB')
        
#         image_patches = create_patches(image, patch_size)
        
#         save_patches(image_patches, base_name, output_image_folder, 'patch')
