import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import UNet
#from datasets.Geometry_dataset import CustomDataset
#from datasets.OneFeature_dataset import CustomDataset
from datasets.WireCheck_dataset import CustomDataset
from utils.heatmap_utils import show_masks_pred1, show_masks_pred, save_valuation
from torchvision.transforms import v2 
import seaborn as sns
import matplotlib.pyplot as plt

def show_predictions(images, masks, preds, idx):
    """
    Zeigt die tatsächlichen Masken und die vorhergesagten Masken für eine gegebene Index an.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(3):
        # Original image
        img = images[idx + i].cpu().numpy().transpose((1, 2, 0))
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Ground truth mask
        mask = masks[idx + i].cpu().squeeze().numpy()
        sns.heatmap(mask, ax=axes[i, 1], cmap='viridis')
        axes[i, 1].set_title('Original Mask')
        axes[i, 1].axis('off')

        # Predicted mask
        pred = preds[idx + i].cpu().squeeze().numpy()
        sns.heatmap(pred, ax=axes[i, 2], cmap='viridis')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

def test(UNet):
    num_class = 1
    #num_class = 6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = UNet(num_class).to(device)
    model.load_state_dict(torch.load('test_RetinaVessel.pth', map_location=device))
    model.eval()

    # trans = v2.Compose([
    #         v2.ToPureTensor(),
    #         v2.ToDtype(torch.float32, scale=True),
    #         #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ])
    
    # trans_mask = v2.Compose([
    #     v2.ToPureTensor(),
    # ])

    # # Create another simulation dataset for test
    # test_dataset = CustomDataset('data', image_transform=trans, mask_transform=trans_mask, count=3)
    # test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=0)
    
    transformations = v2.Compose([
            v2.RandomEqualize(p=1.0),
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),
            #v2.Normalize(mean=mean, std=std)
        ])

    test_dataset = CustomDataset('data_modified/RetinaVessel/train', transform=transformations, count=3)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=0)

    images, masks_tensor = next(iter(test_loader))
    images = images.to(device)
    masks_tensor = masks_tensor.to(device)

    pred = model(images)
    pred = F.sigmoid(pred)
    max = pred.max()
    pred = pred.data.cpu()#.numpy()
    print(pred.shape)
    print(images.shape)
    print(masks_tensor.shape)


    show_predictions(images, masks_tensor, pred, 0)

    #show_masks_pred(mask=masks_tensor,pred=pred)
    #save_valuation(images, masks_tensor, pred)



if __name__ == '__main__':
    try:
        test(UNet)
    except Exception as e:
        print(f"An error occurred: {e}")