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
    plt.figure(figsize=(10, 10))
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
    num_images = pred.shape[0]
    num_masks = pred.shape[1]

    plt.figure(figsize=(20, 20))
    
    # Plot each predicted mask and ground truth mask heatmap
    for i in range(num_masks):
        plt.subplot(2, num_masks, i + 1)
        plt.title(f"Mask {i+1} Prediction")
        sns.heatmap(pred[0, i].cpu().detach().numpy(), cmap='viridis', cbar=True)

        plt.subplot(2, num_masks, num_masks + i + 1)
        plt.title(f"Mask {i+1} Ground Truth")
        sns.heatmap(masks_tensor[0, i].cpu().detach().numpy(), cmap='viridis', cbar=True)
    
    plt.show()

def plot_prediction_heatmaps(pred):
    """
    Plots the prediction heatmaps.

    Parameters:
    - pred (tensor): The predicted masks.
    """
    num_images = pred.shape[0]
    num_masks = pred.shape[1]

    for img_idx in range(num_images):
        fig, axes = plt.subplots(1, num_masks, figsize=(20, 5))
        
        for mask_idx in range(num_masks):
            ax = axes[mask_idx] if num_masks > 1 else axes
            ax.set_title(f"Image {img_idx+1} - Mask {mask_idx+1} Prediction")
            sns.heatmap(pred[img_idx, mask_idx].cpu().detach().numpy(), cmap='viridis', cbar=True, ax=ax)
        
        plt.tight_layout()
        plt.show()

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

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp

def create_combined_mask(masks_tensor):
    """
    Creates a combined mask from individual masks.

    Parameters:
    - masks_tensor (tensor): The tensor containing individual masks with shape [batch_size, num_masks, H, W].

    Returns:
    - combined_mask (tensor): The combined mask with shape [batch_size, H, W].
    """
    batch_size, num_masks, H, W = masks_tensor.shape
    combined_mask = torch.zeros((batch_size, H, W), device=masks_tensor.device)

    for i in range(num_masks):
        combined_mask[masks_tensor[:, i, :, :] > 0.5] = i + 1

    return combined_mask

def visualize_combined_mask(masks_tensor,pred):
    """
    Visualizes the combined mask and individual masks.

    Parameters:
    - masks_tensor (tensor): The tensor containing individual masks.
    - combined_mask (tensor): The combined mask.
    """
    batch_size = masks_tensor.shape[0]

    for img_idx in range(batch_size):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot the individual masks
        ax = axes[0]
        ax.set_title(f"Image {img_idx+1} - Individual Masks")
        for i in range(masks_tensor.shape[1]):
            mask = masks_tensor[img_idx, i].cpu().detach().numpy()
            ax.imshow(mask, alpha=0.5, cmap='viridis')

        # Plot the individual masks
        ax = axes[1]
        ax.set_title(f"Image {img_idx+1} - Prediction Masks")
        for i in range(pred.shape[1]):
            pred_show = pred[img_idx, i].cpu().detach().numpy()
            ax.imshow(pred_show, alpha=0.5, cmap='viridis')
        
        plt.tight_layout()
        plt.show()

def show_masks_pred(mask, pred):
    # Wählen Sie den ersten Batch aus und konvertieren Sie ihn in NumPy (auf CPU kopieren)
    true_masks = mask[0].cpu().numpy()
    pred_masks = pred[0].cpu().numpy()

    # Anzahl der Bilder
    num_images = true_masks.shape[0]

    # # Berechne globale minimale und maximale Werte
    # min_val = min([true_masks[i].min() for i in range(num_images)] + [pred_masks[i].min() for i in range(num_images)])
    # max_val = max([true_masks[i].max() for i in range(num_images)] + [pred_masks[i].max() for i in range(num_images)])

    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))

    for i in range(num_images):
        image = true_masks[i]
        axes[0, i].imshow(image, cmap='gray', vmin=0, vmax=0.1)
        axes[0, i].axis('off')

    for i in range(num_images):
        image = pred_masks[i]
        sns.heatmap(image, ax=axes[1, i], cmap='viridis', cbar=True, vmin=0, vmax=0.1)
        axes[1, i].axis('off')

    plt.show()