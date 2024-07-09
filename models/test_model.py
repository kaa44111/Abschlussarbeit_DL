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
from datasets.Geometry_dataset import CustomDataset
#from datasets.OneFeature_dataset import CustomDataset
from utils.heatmap_utils import show_masks_pred1, visualize_predictions, show_masks_pred
from torchvision.transforms import v2 


def test(UNet):
    #num_class = 1
    num_class = 6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = UNet(num_class).to(device)
    model.load_state_dict(torch.load('trained/better_normalized_data.pth', map_location=device))
    model.eval()

    trans = v2.Compose([
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    trans_mask = v2.Compose([
        v2.ToPureTensor(),
    ])

    # Create another simulation dataset for test
    test_dataset = CustomDataset('data', image_transform=trans, mask_transform=trans_mask, count=3)
    #test_dataset = CustomDataset('data/circle_data/val', transform=trans, count=3)
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

    show_masks_pred(mask=masks_tensor,pred=pred)
   # visualize_predictions(pred, masks_tensor)



if __name__ == '__main__':
    try:
        test(UNet)
    except Exception as e:
        print(f"An error occurred: {e}")