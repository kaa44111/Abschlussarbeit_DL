import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from PIL import Image

def plot_heatmap(data, title=None, cmap='viridis'):
    """
    Plots a heatmap using seaborn.

    Parameters:
    - data (2D array): The data to be visualized as a heatmap.
    - title (str): Optional title for the heatmap.
    - cmap (str): Optional colormap for the heatmap.
    """
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(data, cmap=cmap, cbar=True)
    if title:
        ax.set_title(title)
    plt.show()

#Festkodiert für Geometry_dataset!
def visualize_predictions(pred, masks_tensor):
    """
    Visualizes the prediction heatmaps and ground truth mask heatmaps.

    Parameters:
    - pred (tensor): The predicted masks.
    - masks_tensor (tensor): The ground truth masks.
    """
    num_masks = masks_tensor.shape[1]

    plt.figure(figsize=(20, 10))
    
    # Plot each predicted mask and ground truth mask heatmap
    for i in range(num_masks):
        plt.subplot(2, num_masks, i + 1)
        plt.title(f"Mask {i+1} Prediction")
        sns.heatmap(pred[0, i].cpu().detach().numpy(), cmap='viridis', cbar=True)

        plt.subplot(2, num_masks, num_masks + i + 1)
        plt.title(f"Mask {i+1} Ground Truth")
        sns.heatmap(masks_tensor[0, i].cpu().detach().numpy(), cmap='viridis', cbar=True)
    
    plt.show()
