import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.utils
from collections import defaultdict
import time
import copy
from PIL import Image
import numpy as np


def renormalize(tensor):
        minFrom= tensor.min()
        maxFrom= tensor.max()
        minTo = 0
        maxTo=1
        return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))

def save_images_und_masks(inputs, label):
    # Verarbeite das Eingabebild
    first_image = inputs[0]  # Das erste Bild im Batch
    first_image_2d = first_image.permute(1, 2, 0)  # Ändert die Dimensionen von [3, 192, 192] zu [192, 192, 3]
    
    # Bild normalisieren und speichern
    tensorNeu = renormalize(first_image_2d)
    image_array = (tensorNeu * 255).cpu().detach().numpy().astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save('test_trainloop/images/a.png')

    # Verarbeite die Masken
    first_label = label[0]  # Die Masken des ersten Bildes im Batch
    num_masks = first_label.shape[0]  # Anzahl der Masken

    for i in range(num_masks):
        mask = first_label[i]
        mask_array = mask.cpu().detach().numpy()
        
        # Normalisierung der Maske, falls die Werte nicht im Bereich [0, 1] sind
        mask_array1 = renormalize(mask_array)
        print(f"Mask {i} min: {mask_array1.min()}, max: {mask_array1.max()}")
        
        mask_array = (mask_array1 * 255).astype(np.uint8)  # Konvertiere zum uint8 Format
        mask_image = Image.fromarray(mask_array, mode='L')  # Mode 'L' für Graustufenbilder
        mask_image.save(f'test_trainloop/masks/mask_{i}.png')


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



def train_model(model,dataloaders, optimizer, scheduler, num_epochs=25):
    #dataloaders,_ = get_dataloaders('data_modified/RetinaVessel/train')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
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

                # # Debugging-Ausgaben
                # print(f"Inputs shape: {inputs.shape}, dtype: {inputs.dtype}")
                # print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
                # print(f"Max input value: {inputs.max()}, Min input value: {inputs.min()}")
                # print(f"Max label value: {labels.max()}, Min label value: {labels.min()}")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    #save_images_und_masks(inputs, labels)

                    outputs = model(inputs)

                    # print(f"Outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
                    # print(f"Max output value: {outputs.max()}, Min output value: {outputs.min()}")

                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

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
    return model

def run(UNet,dataloader,dataset_name,save_name):
    num_class = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(num_class).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    start = time.time()
    model = train_model(model,dataloader, optimizer_ft, exp_lr_scheduler, num_epochs=30)
    end = time.time()

    # Verstrichene Zeit in Sekunden
    elapsed_time_s = end - start
    elapsed_time_min = elapsed_time_s / 60

    print(f"Elapsed time: {elapsed_time_min:.2f} minutes")
    print("\n")

    results_dir = os.path.join('train/results',dataset_name)
    save_dir = f"{results_dir}/{save_name}.pth"

    # Speichern des trainierten Modells
    torch.save(model.state_dict(), save_dir)
    print(f"Model saved to {save_dir}")
    

# if __name__ == '__main__':
#      try:
        
#      except Exception as e:
#         print(f"An error occurred: {e}")

############
# Geometry_dataset
# num_class = 6
# epochs = 30
# Images as Grey value : 10.24 minutes
#___________________________
# Images as RGB : 12.18 minutes

###########
# Geometry_dataset
# num_class = 6
# epochs = 75
# Images as Grey value : 25.61 minutes
# LR 1.0000000000000002e-06
# Best val loss: 0.118130
#_________________________
# Images as RGB : 35.16 minutes
# LR 1.0000000000000002e-06
# Best val loss: 0.147360

############
# num_class = 6
# device = gpu
# lr=1e-4
# epochs=35
# batchsize=30
# trainset = 100, valset = 20