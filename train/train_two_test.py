import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from collections import defaultdict
import time
import copy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datasets.OneFeature import get_dataloaders
from models.UNetBatchNorm import UNetBatchNorm
from models.UNet import UNet

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_metrics(metrics1, metrics2, metric_name):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics1[metric_name], label='UNet')
    plt.plot(metrics2[metric_name], label='UNetMaxPool')
    plt.title(f'{metric_name} over epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(f'{metric_name}_comparison.png')
    plt.close()

def measure_inference_time(model, input_tensor, num_iterations=100):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    return avg_time

def train_model(model, dataloaders, optimizer, scheduler, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    history = defaultdict(list)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            epoch_dice = 1 - (metrics['dice'] / epoch_samples)  # Dice-Koeffizient
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_dice'].append(epoch_dice)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def run():
    root_dir = 'data_modified/RetinaVessel/train'
    dataloaders,_ = get_dataloaders(root_dir=root_dir)
    
    num_class = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model1 = UNet(num_class)  # Originalmodell
    model2 = UNetBatchNorm(num_class)  # Modell mit Batch-Normalisierung
    
    # Anzahl der Parameter
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)
    print(f"UNet parameters: {params1}")
    print(f"UNetBatchNorm parameters: {params2}")

    optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, model1.parameters()), lr=1e-4)
    scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=30, gamma=0.1)

    optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=1e-4)
    scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=30, gamma=0.1)
    
    # Training
    model1, history1 = train_model(model1, dataloaders, optimizer1, scheduler1, num_epochs=30)
    model2, history2 = train_model(model2, dataloaders, optimizer2, scheduler2, num_epochs=30)
    
    # Vergleich der Metriken
    plot_metrics(history1, history2, 'val_loss')
    plot_metrics(history1, history2, 'val_dice')
    
    # Inferenzzeit messen
    sample_input = next(iter(dataloaders['val']))[0][:1].to(device)
    inference_time1 = measure_inference_time(model1, sample_input)
    inference_time2 = measure_inference_time(model2, sample_input)
    print(f"UNet inference time: {inference_time1:.4f} seconds")
    print(f"UNetBatchNorm inference time: {inference_time2:.4f} seconds")
    
    # Speichern des UNet Modells
    torch.save(model1.state_dict(), 'UNet_RetinaVessel.pth')
    print("Model saved to UNet_RetinaVessel.pth")

    # Speichern des UNetBatchNorm Modells
    torch.save(model2.state_dict(), 'UNetBatchNorm_RetinaVessel.pth')
    print("Model saved to UNetBatchNorm_RetinaVessel.pth")

if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}")

##################
# Epochs : 30
# Batch : 15
# UNet parameters: 31031745
# UNetBatchNorm parameters: 31043521