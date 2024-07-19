import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
import torch.utils
from torchvision.transforms import v2

#from datasets.MultipleFeature import get_data_loaders
from datasets.OneFeature import get_dataloaders

#Prepare
from prepare.prepare_binning import prepare_binning
from prepare.prepare_patches import prepare_patches
from prepare.prepare_both import process_images

#Train
from train.train import run
from train.train_compare import run_compare

#Test
from test_models.ImageOnly_test import test
from test_models.test_different_models import test_compare

#Falls man die Bilder Normalisiern will
from utils.data_utils import compute_mean_std, show_image_and_mask

#Modelle:
from models.UNet import UNet
from models.UNetBatchNorm import UNetBatchNorm
from models.UNetMaxPool import UNetMaxPool


if __name__ == '__main__':
     try:
        #Dataset Informations (root_dir, dataset_name)
        '''
        Default dataset_name = data/{dataset_name}
        '''
        root_dir= 'data/Dichtflächen'
        dataset_name = 'Dichtflächen'

        #Prepare Dataset (downsampe, batch, both)
        '''
        Default downsample : scale_factor = 2
        Default patch:  patch_size= 200
        '''
        #train_dir = prepare_binning(root_dir,5,dataset_name)
        #train_dir = prepare_patches(root_dir=root_dir,dataset_name=dataset_name)
        train_dir = process_images(root_dir,dataset_name,2,200)
        #Get Dataloader
        '''
        Default Transform: ToPureTensor(), ToDtype(torch.float32, scale=True)
        Default batch_size : 15
        Default split_size : 0.8
        '''
        dataloader,_ = get_dataloaders(root_dir=train_dir)
        batch = next(iter(dataloader['train']))
        images,masks = batch
        print(images.shape)
        print(masks.shape)
        print(f"First image min: {images[0].min()}, max: {images[0].max()}")
        print(f"First mask min: {masks[0].min()}, max: {masks[0].max()}") 

        

        #_____________________________________________________________

        # ####Training für ein Modell Starten
        # print("Train Model with Dichtflächen Dataset:")
        # run(UNet,dataloader,dataset_name,save_name)

        # results_dir = os.path.join('train/results',dataset_name)
        # trained_model = f"{results_dir}/{save_name}.pth"

        # ####Testen für das antrainerte Modell Starten
        # print("Test Results:")
        # test(UNet=UNet,test_dir=test_dir,trained_path=trained_model)

        #______________________________________________________________

        # #####Training für alle Modelle Starten
        # run_compare(dataloader,dataset_name)

        # #####Test für alle trainerte Modelle
        # test_compare(test_dir, dataset_name, UNet, UNetMaxPool, UNetBatchNorm)
        
     except Exception as e:
        print(f"An error occurred: {e}")
