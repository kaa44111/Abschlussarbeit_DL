import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_utils import MAPPING,custom_collate_fn
from model import UNet
from datasets.Geometry_dataset import CustomDataset
#from datasets.OneFeature_dataset import CustomDataset
from utils.data_utils import BinningTransform
from utils.heatmap_utils import visualize_predictions
from torchvision.transforms import v2 
import numpy as np
from PIL import Image


def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)
def test(UNet):
    #num_class = 1
    num_class = 6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = UNet(num_class).to(device)
    model.load_state_dict(torch.load('trained/new.pth', map_location=device))
    model.eval()

    trans = v2.Compose([
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    # Create another simulation dataset for test
    test_dataset = CustomDataset('data', transform=trans, count=3)
    #test_dataset = CustomDataset('data/circle_data/val', transform=trans)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

    images, masks_tensor = next(iter(test_loader))
    images = images.to(device)
    masks_tensor = masks_tensor.to(device)

    pred = model(images)
    pred = F.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    print(pred.shape)

    pred_rgb = [masks_to_colorimg(x) for x in pred]
    
    i = 0
    for prd in pred_rgb:
        i += 1
        im = Image.fromarray(prd)
        im.save("data\\geometry_shapes\\validate\\" + str(i) + ".png")    
    
    # Visualisieren der Vorhersagen und Heatmaps
    #visualize_predictions(pred, masks_tensor)

if __name__ == '__main__':
    try:
        test(UNet)
    except Exception as e:
        print(f"An error occurred: {e}")