import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.utils
from collections import defaultdict
import time
import copy
from model import UNet
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
#from datasets.OneFeature_dataset import get_dataloaders
from datasets.Geometry_dataset import get_data_loaders


def renormalize(tensor):
        minFrom= tensor.min()
        maxFrom= tensor.max()
        minTo = 0
        maxTo=1
        return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))

def save_images_und_masks(inputs, label):
    # tensor = torch.split(inputs)

    # test = torch.squeeze(tensor[0], 0) 
    # tsh = test.shape
    # tensor2 = torch.split(test,1)

    # sh = tensor2[0].shape
    # test2 = torch.squeeze(tensor2[0], 0) 
    # tsh2 = test2.shape

    # tensorNeu = renormalize(test2)

                    
    # minFromNeu= tensorNeu.min()
    # maxFromNeu= tensorNeu.max()

    # a=tensorNeu.cpu().detach().numpy().astype(np.uint8)
    # image = Image.fromarray(a)

    # # Bild speichern
    # image.save('test_trainloop/images/a.png')

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

        print(f"Mask {i} min: {mask_array.min()}, max: {mask_array.max()}")
        
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



def train_model(model, optimizer, scheduler, num_epochs=25):
    dataloaders,_ = get_data_loaders()
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
                
                
                testInputs = inputs.data.cpu().numpy()
                testLabels = labels.data.cpu().numpy()

                

                testInputs2 = testInputs[0][0]
                testLabels2 = testLabels[0][0]
                test = testInputs2.max()
                test1 = testLabels2.max()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    #save_images_und_masks(inputs, labels)

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

def run(UNet):
    num_class = 6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(num_class).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=30)

    # Speichern des trainierten Modells
    torch.save(model.state_dict(), 'trained/normalized_data.pth')
    print("Model saved to trained/normalized_data.pth")

if __name__ == '__main__':
     try:
         run(UNet)
     except Exception as e:
         print(f"An error occurred: {e}")

############

# num_class = 6
# device = gpu
# lr=1e-4
# epochs=35
# batchsize=30
# trainset = 100, valset = 20