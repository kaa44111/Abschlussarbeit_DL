import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
    
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from torchvision.transforms import v2
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from models.UNet import UNet
from models.UNetBatchNorm import UNetBatchNorm
import matplotlib
matplotlib.use('TkAgg')  # Backend auf TkAgg umstellen


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
            image = v2.pil_to_tensor(image).float() / 255.0

            if self.transform:
                image = self.transform(image)

            return image, self.image_files[idx]  # Return the filename as well

        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None

def save_heatmap(pred, filename, save_dir):
    plt.figure(figsize=(10, 10))
    sns.heatmap(pred, cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'heatmap_{filename}'), bbox_inches='tight', pad_inches=0)
    plt.close()

def test(UNet, UNetMaxPool, UNetBatchNorm):
    num_class = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model1 = UNet(num_class).to(device)  # Original model
    model2 = UNetMaxPool(num_class).to(device)  # Model without MaxPool
    model3 = UNetBatchNorm(num_class).to(device)  # Model with Batch Normalization
    
    model1.load_state_dict(torch.load('UNet_RetinaVessel.pth', map_location=device))
    model2.load_state_dict(torch.load('UNetMaxPool_RetinaVessel.pth', map_location=device))
    model3.load_state_dict(torch.load('UNetBatchNorm_RetinaVessel.pth', map_location=device))
    
    model1.eval()
    model2.eval()
    model3.eval()
    
    transformations = v2.Compose([
            v2.RandomEqualize(p=1.0),
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),
    ])

    test_dataset = ImageOnlyDataset('data_modified/RetinaVessel/test', transform=transformations)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create directories for saving heatmaps
    os.makedirs('evaluate/heatmaps_UNet', exist_ok=True)
    os.makedirs('evaluate/heatmaps_UNetMaxPool', exist_ok=True)
    os.makedirs('evaluate/heatmaps_UNetBatchNorm', exist_ok=True)

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)

            # Predictions for UNet
            pred1 = model1(images)
            pred1 = F.sigmoid(pred1)
            pred1 = pred1.squeeze().cpu().numpy()

            # Predictions for UNetMaxPool
            pred2 = model2(images)
            pred2 = F.sigmoid(pred2)
            pred2 = pred2.squeeze().cpu().numpy()

            # Predictions for UNetBatchNorm
            pred3 = model3(images)
            pred3 = F.sigmoid(pred3)
            pred3 = pred3.squeeze().cpu().numpy()

            # Save heatmaps
            save_heatmap(pred1, filenames[0], 'heatmaps_UNet')
            save_heatmap(pred2, filenames[0], 'heatmaps_UNetMaxPool')
            save_heatmap(pred3, filenames[0], 'heatmaps_UNetBatchNorm')

    print("Heatmaps saved in 'heatmaps_UNet', 'heatmaps_UNetMaxPool', and 'heatmaps_UNetBatchNorm' folders.")

if __name__ == '__main__':
    try:
        from models.UNet import UNet
        from models.UNetMaxPool import UNetMaxPool
        from models.UNetBatchNorm import UNetBatchNorm
        test(UNet, UNetMaxPool, UNetBatchNorm)
    except Exception as e:
        print(f"An error occurred: {e}")