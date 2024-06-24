import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from collections import defaultdict
import time
import copy
from model import UNet
from datasets.Geometry_dataset import get_dataloaders, CustomDataset
from utils.data_utils import BinningTransform
from utils.heatmap_utils import visualize_colored_heatmaps
from torchvision.transforms import v2 
import datasets.Geometry_dataset
import seaborn as sns
import matplotlib as plt



def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, scheduler, num_epochs):
    dataloaders = get_dataloaders()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        for phase in ['train', 'val']:


            metrics = defaultdict(float)
            epoch_samples = 0

            for i, (inputs, labels, _) in enumerate(dataloaders[phase]):
                if i > 0:  # Nur einen Batch pro Epoche verwenden
                    break
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

                if phase == 'val':
                    preds = torch.sigmoid(outputs)
                    preds = (preds > 0.5).float()
                    visualize_colored_heatmaps(inputs, preds, labels)
            
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()
            else:
                model.eval()

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    return model

def run():
    num_class = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = UNet(num_class).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=1)

    model.eval()

    trans = v2.Compose([
        v2.ToPureTensor(),
        BinningTransform(bin_size=2),
        v2.ToDtype(torch.float32, scale=True),
    ])

    test_dataset = CustomDataset('data', transform=trans, mapping=None)
    test_loader = datasets.Geometry_dataset.DataLoader(test_dataset, batch_size=1, shuffle=True)

    inputs, labels, _ = next(iter(test_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    pred = model(inputs)
    pred = F.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    plt.show()
   # visualize_colored_heatmaps(inputs, pred, labels)
    print(pred.shape)

if __name__ == "__main__":
    try:
        run()

    except Exception as e:
        print(f"An error occurred: {e}")


