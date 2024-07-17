import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import copy
import time
from collections import defaultdict
import matplotlib.pyplot as plt

from datasets.OneFeature import get_dataloaders
from models.UNetBatchNorm import UNetBatchNorm
from models.UNet import UNet
from models.UNetMaxPool import UNetMaxPool

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_metrics(metrics1, metrics2, metrics3, metric_name):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics1[metric_name], label='UNet')
    plt.plot(metrics2[metric_name], label='UNetMaxPool')
    plt.plot(metrics3[metric_name], label='UNetBatchNorm')
    plt.title(f'{metric_name} over epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(f'train/results/compare_results/{metric_name}_comparison.png')
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
    start_time = time.time()  # Startzeit des Trainings

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()  # lr_scheduler.step() nach optimizer.step() aufrufen

                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            epoch_dice = 1 - (metrics['dice'] / epoch_samples)  # Dice-Koeffizient

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_dice'].append(epoch_dice)

            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed1 = time.time() - since
        time_elapsed = time.time() - start_time
        print('{:.0f}m {:.0f}s'.format(time_elapsed1 // 60, time_elapsed1 % 60))

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    time_elapsed_min = time_elapsed / 60
    model.load_state_dict(best_model_wts)
    return model, history, time_elapsed_min

def run(train_dir,dataset_name):
    #root_dir = 'data_modified/RetinaVessel/train'
    dataloaders, _ = get_dataloaders(root_dir=train_dir,dataset_name=dataset_name)
    
    num_class = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model1 = UNet(num_class)  # Originalmodell
    model2 = UNetMaxPool(num_class)  # Modell ohne die MaxPool Schicht
    model3 = UNetBatchNorm(num_class)  # Modell mit BatchNorm Schichten

    print("#######################################")
    print("Start training Original UNet Model:")    
    # Training des Originalmodells
    optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, model1.parameters()), lr=1e-4)
    scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=30, gamma=0.1)
    model1, history1, time1 = train_model(model1, dataloaders, optimizer1, scheduler1, num_epochs=30)

    print("#######################################")
    print("Start training UNet Model without MaxPool:")
    # Training des Modells ohne MaxPool
    optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=1e-4)
    scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=30, gamma=0.1)
    model2, history2, time2 = train_model(model2, dataloaders, optimizer2, scheduler2, num_epochs=30)

    print("#######################################")
    print("Start training UNet Model with BatchNorm:")
    # Training des Modells mit BatchNorm
    optimizer3 = optim.Adam(filter(lambda p: p.requires_grad, model3.parameters()), lr=1e-4)
    scheduler3 = lr_scheduler.StepLR(optimizer3, step_size=30, gamma=0.1)
    model3, history3, time3 = train_model(model3, dataloaders, optimizer3, scheduler3, num_epochs=30)

    # Vergleich der Metriken
    plot_metrics(history1, history2, history3, 'val_loss')
    plot_metrics(history1, history2, history3, 'val_dice')

    print("\n")
    # Vergleich der Trainingszeiten
    print(f"UNet training time: {time1:.2f} seconds")
    print(f"UNetMaxPool training time: {time2:.2f} seconds")
    print(f"UNetBatchNorm training time: {time3:.2f} seconds")
    print("\n")

    # Inferenzzeit messen
    '''Zeit, die ein Modell benötigt, um eine Vorhersage für ein einzelnes Eingabebild zu machen'''
    sample_input = next(iter(dataloaders['val']))[0][:1].to(device)
    inference_time1 = measure_inference_time(model1, sample_input)
    inference_time2 = measure_inference_time(model2, sample_input)
    inference_time3 = measure_inference_time(model3, sample_input)
    print(f"UNet inference time: {inference_time1:.4f} min")
    print(f"UNetMaxPool inference time: {inference_time2:.4f} min")
    print(f"UNetBatchNorm inference time: {inference_time3:.4f} min")
    print("\n")

    # Anzahl der Parameter
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)
    params3 = count_parameters(model3)
    print(f"UNet parameters: {params1}")
    print(f"UNetMaxPool parameters: {params2}")
    print(f"UNetBatchNorm parameters: {params3}")
    print("\n")

    # Speichern der trainierten Modelle
    compare_results = os.path.join('train/results/compare_results',dataset_name)
    torch.save(model1.state_dict(), f"{compare_results}/UNet.pth")
    torch.save(model2.state_dict(), f"{compare_results}/UNetBatchNorm.pth")
    torch.save(model3.state_dict(), f"{compare_results}/UNetBatchNorm.pth")
    print("Models saved to disk")

if __name__ == '__main__':
    try:
        train_dir = 'data_modified/RetinaVessel/train'
        dataset_name = 'RetinaVessel'
        run(train_dir,dataset_name)
    except Exception as e:
        print(f"An error occurred: {e}")


# UNet Best val loss: 0.628031
# UNetMaxPool Best val loss: 0.633133
# UNetBatchNorm Best val loss: 0.467543


# UNet training time: 4m 18s
# UNetMaxPool training time: 1m 12s
# UNetBatchNorm training time: 20m 8s


# UNet inference time: 0.0410 seconds
# UNetMaxPool inference time: 0.0428 seconds
# UNetBatchNorm inference time: 0.0566 seconds


# UNet parameters: 31031745
# UNetMaxPool parameters: 31031745
# UNetBatchNorm parameters: 31043521