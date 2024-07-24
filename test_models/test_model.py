import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision.transforms import v2 
# import seaborn as sns
# import matplotlib.pyplot as plt

# from datasets.OneFeature import CustomDataset
# from utils.heatmap_utils import show_masks_pred1, show_masks_pred, save_valuation


# def show_predictions(images, masks, preds, idx):
#     """
#     Zeigt die tatsächlichen Masken und die vorhergesagten Masken für eine gegebene Index an.
#     """
#     fig, axes = plt.subplots(3, 3, figsize=(15, 15))

#     for i in range(3):
#         # Original image
#         img = images[idx + i].cpu().numpy().transpose((1, 2, 0))
#         axes[i, 0].imshow(img)
#         axes[i, 0].set_title('Original Image')
#         axes[i, 0].axis('off')

#         # Ground truth mask
#         mask = masks[idx + i].cpu().squeeze().numpy()
#         sns.heatmap(mask, ax=axes[i, 1], cmap='viridis')
#         axes[i, 1].set_title('Original Mask')
#         axes[i, 1].axis('off')

#         # Predicted mask
#         pred = preds[idx + i].cpu().squeeze().numpy()
#         sns.heatmap(pred, ax=axes[i, 2], cmap='viridis')
#         axes[i, 2].set_title('Predicted Mask')
#         axes[i, 2].axis('off')

#     plt.tight_layout()
#     plt.show()

# def test(UNet,test_dir,test_trained_model,transformations,dataset_name=None):
#     num_class = 1
#     #num_class = 6
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     model = UNet(num_class).to(device)
#     model.load_state_dict(torch.load(test_trained_model, map_location=device))
#     model.eval()

#     test_dataset = CustomDataset(test_dir, dataset_name=dataset_name, transform=transformations, count=3)
#     test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=0)

#     images, masks_tensor = next(iter(test_loader))
#     images = images.to(device)
#     masks_tensor = masks_tensor.to(device)

#     pred = model(images)
#     pred = F.sigmoid(pred)
#     max = pred.max()
#     pred = pred.data.cpu()#.numpy()
#     print(pred.shape)
#     print(images.shape)
#     print(masks_tensor.shape)


#     show_predictions(images, masks_tensor, pred, 0)

#     #show_masks_pred(mask=masks_tensor,pred=pred)
#     #save_valuation(images, masks_tensor, pred)



# if __name__ == '__main__':
#     try:
#         from models.UNet import UNet
#         #from models.UNetBatchNorm import UNetBatchNorm
#         #from models.UNetMaxPool import UNetMaxPool

#         test_dir = 'data/Ölflecken'
#         dataset_name = 'Ölflecken'
#         test_trained_model = 'train/results/Ölflecken/test_train.pth'

#         test(UNet,test_dir,dataset_name,test_trained_model)

#     except Exception as e:
#         print(f"An error occurred: {e}")

#___________________________________________________________
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from collections import defaultdict
from train.train import print_metrics, calc_loss
from torchvision.transforms import v2


class ImageOnlyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(self.image_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))

        print(f"Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.image_dir, self.image_files[idx])
            image = Image.open(img_name).convert('RGB')
            image = v2.functional.pil_to_tensor(image).float() / 255.0

            if self.transform:
                image = self.transform(image)

            return image

        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None

def load_model(model_class, checkpoint_path, num_class=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_class(num_class)
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()
    return model

def save_predictions(model, dataloader, output_folder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_folder, exist_ok=True)

    for i, inputs in enumerate(dataloader):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()  # Apply a threshold

        preds = preds.cpu().numpy()

        for j in range(preds.shape[0]):
            pred_img = preds[j][0] * 255  # Assuming single channel output
            pred_img = Image.fromarray(pred_img.astype(np.uint8))
            pred_img.save(os.path.join(output_folder, f"prediction_{i * preds.shape[0] + j}.png"))

    print(f"Predictions saved to {output_folder}")

def test_model(UNet, test_dataloader, checkpoint_path, output_folder):
    # Load the model
    model = load_model(UNet, checkpoint_path)

    # Save predictions
    print("Saving predictions...")
    save_predictions(model, test_dataloader, output_folder)

if __name__ == '__main__':

    try:
        from models.UNetBatchNorm import UNetBatchNorm
        test_dir = 'data/data_modified/Dichtflächen/patched/test'
    
        transformations = v2.Compose([
            #v2.RandomEqualize(p=1.0),
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),
            #v2.Normalize(mean=mean, std=std)
            ])

        test_dataset = ImageOnlyDataset(test_dir, transform=transformations)
        test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True, num_workers=0)

        test_model(UNetBatchNorm, test_loader, 'train/results/Dichtflächen/test_UNetBatchNorm.pth', 'test_models/evaluate/Dichfläche')

    except Exception as e:
        print(f"An error occurred: {e}")
# Example usage:
# Define the test dataloader
# test_dataloader = DataLoader(...)

# test_model(UNet, test_dataloader, 'path_to_trained_model.pth', 'output_folder_for_predictions')