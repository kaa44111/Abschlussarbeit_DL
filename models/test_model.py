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
from utils.data_utils import BinningTransform
from utils.heatmap_utils import visualize_predictions
from torchvision.transforms import v2 

def test(UNet):
    num_class = 6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = UNet(num_class).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    trans = v2.Compose([
        v2.ToPureTensor(),
        BinningTransform(bin_size=2),  # Beispiel für Binning mit bin_size 2
        v2.ToDtype(torch.float32, scale=True),
    ])

    # Create another simulation dataset for test
    test_dataset = CustomDataset('data', transform=trans, mapping=MAPPING)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    images, _ , masks_tensor = next(iter(test_loader))
    images = images.to(device)
    masks_tensor = masks_tensor.to(device)

    pred = model(images)
    pred = F.sigmoid(pred)

    # Visualisieren der Vorhersagen und Heatmaps
    visualize_predictions(pred, masks_tensor)

# if __name__ == '__main__':
#     try:
#         test(UNet)
#     except Exception as e:
#         print(f"An error occurred: {e}")