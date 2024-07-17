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
from torchvision import tv_tensors
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
            image = tv_tensors.Image(image)

            if self.transform:
                image = self.transform(image)

            return image, self.image_files[idx]  # Return the filename as well

        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None

def save_combined_image(image, pred1, pred2, pred3, filename, save_dir):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axs[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Prediction heatmap UNet
    sns.heatmap(pred1, cmap='viridis', ax=axs[1], cbar=False)
    axs[1].set_title('UNet')
    axs[1].axis('off')
    
    # Prediction heatmap UNetMaxPool
    sns.heatmap(pred2, cmap='viridis', ax=axs[2], cbar=False)
    axs[2].set_title('UNetMaxPool')
    axs[2].axis('off')
    
    # Prediction heatmap UNetBatchNorm
    sns.heatmap(pred3, cmap='viridis', ax=axs[3], cbar=False)
    axs[3].set_title('UNetBatchNorm')
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'combined_{filename}'), bbox_inches='tight', pad_inches=0)
    plt.close()

def test_compare(test_dir, dataset_name, UNet, UNetMaxPool, UNetBatchNorm):
    num_class = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model1 = UNet(num_class).to(device)  # Original model
    model2 = UNetMaxPool(num_class).to(device)  # Model without MaxPool
    model3 = UNetBatchNorm(num_class).to(device)  # Model with Batch Normalization
    
    compare_results = os.path.join('train/results/compare_results', dataset_name)
    model1.load_state_dict(torch.load(f"{compare_results}/UNet.pth", map_location=device))
    model2.load_state_dict(torch.load(f"{compare_results}/UNetMaxPool.pth", map_location=device))
    model3.load_state_dict(torch.load(f"{compare_results}/UNetBatchNorm.pth", map_location=device))

    model1.eval()
    model2.eval()
    model3.eval()
    
    transformations = v2.Compose([
            v2.RandomEqualize(p=1.0),
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),
    ])

    test_dataset = ImageOnlyDataset(test_dir, transform=transformations)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    evaluate_dir = os.path.join('test_models/evaluate', dataset_name)
    os.makedirs(evaluate_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, filenames) in enumerate(test_loader):
            if idx >= 4:  # Limit to 4 examples
                break
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

            # Save combined images
            save_combined_image(images[0], pred1, pred2, pred3, filenames[0], evaluate_dir)

    print(f"Combined images saved in '{evaluate_dir}' folder.")

# if __name__ == '__main__':
#     try:
#         from models.UNet import UNet
#         from models.UNetMaxPool import UNetMaxPool
#         from models.UNetBatchNorm import UNetBatchNorm

#         test_dir = 'data_modified/RetinaVessel/test'
#         dataset_name = 'RetinaVessel'

#         test_compare(test_dir, dataset_name, UNet, UNetMaxPool, UNetBatchNorm)
#     except Exception as e:
#         print(f"An error occurred: {e}")