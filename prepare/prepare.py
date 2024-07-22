import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
    
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision import transforms
from datasets.OneFeature import CustomDataset
from utils.data_utils import compute_mean_std_from_dataset, compute_mean_std
from PIL import Image

if __name__ == '__main__':
    try: 
        root_dir = 'data_modified/RetinaVessel/train'
        image_folder = os.path.join(root_dir, 'grabs')  # Verzeichnis mit deinen Bildern

        trans= transforms.Compose([
            transforms.ToTensor(),
            v2.RandomEqualize(p=1.0)
        ])

        dataset=CustomDataset(root_dir=root_dir,transform=trans)
        mean, std = compute_mean_std_from_dataset(dataset)
        print(f"Mean: {mean}")
        print(f"Std: {std}")

        trans= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),
        ])

        dataset1=CustomDataset(root_dir=root_dir,transform=trans)
        image,mask = dataset1[0]
        print(image.min(), image.max())

        # for i in range(76,80):
        #     unnormalized_image,_ = dataset[i]
        #     normalized_image, mask = dataset1[i]
        #     show_normalized_images(unnormalized_image, normalized_image, mask)
        #     #print(mask.min(), mask.max())


    except Exception as e:
        print(f"An error occurred: {e}")

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
#__________________________________________________________________________________________________
