import numpy as np
import matplotlib.pyplot as plt
import torch

def create_heatmap(pred, target):
    """Erstellt eine Heatmap, die die Übereinstimmung zwischen Vorhersagen und Zielwerten zeigt."""
    heatmap = np.zeros_like(pred, dtype=np.float32)
    heatmap[pred == target] = 1  # Richtig klassifizierte Pixel
    return heatmap

def visualize_heatmaps(images, preds, targets):
    """Visualisiert die Bilder, Vorhersagen, Zielwerte und Heatmaps."""
    batch_size = images.shape[0]
    
    for i in range(batch_size):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        pred = preds[i].squeeze().cpu().numpy()
        target = targets[i].squeeze().cpu().numpy()
        heatmap = create_heatmap(pred, target)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img)
        axes[0].set_title('Image')
        
        axes[1].imshow(pred, cmap='gray')
        axes[1].set_title('Prediction')
        
        axes[2].imshow(target, cmap='gray')
        axes[2].set_title('Target')
        
        axes[3].imshow(heatmap, cmap='hot', interpolation='nearest')
        axes[3].set_title('Heatmap')
        
        for ax in axes:
            ax.axis('off')
        
        plt.show()

def create_colored_heatmap(pred, target):
    """Erstellt eine farbige Heatmap, die die Übereinstimmung zwischen Vorhersagen und Zielwerten zeigt."""
    heatmap = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.float32)
    
    # True Positives (richtig als Positiv erkannt) in Grün
    heatmap[(pred == 1) & (target == 1)] = [0, 1, 0]
    
    # True Negatives (richtig als Negativ erkannt) in Schwarz
    heatmap[(pred == 0) & (target == 0)] = [0, 0, 0]
    
    # False Negatives (falsch als Negativ erkannt) in Rot
    heatmap[(pred == 0) & (target == 1)] = [1, 0, 0]
    
    # False Positives (falsch als Positiv erkannt) in Blau
    heatmap[(pred == 1) & (target == 0)] = [0, 0, 1]
    
    return heatmap

def visualize_colored_heatmaps(images, preds, targets):
    """
    Visualisiert die Bilder, Vorhersagen, Zielwerte und Heatmaps. 
    Grün: Richtig klassifizierte Pixel.
    Rot: Falsch negative Pixel (vorhergesagt als negativ, aber tatsächlich positiv).
    Blau: Falsch positive Pixel (vorhergesagt als positiv, aber tatsächlich negativ).

    """
    batch_size = images.shape[0]
    
    for i in range(batch_size):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        pred = preds[i].squeeze().cpu().numpy()
        target = targets[i].squeeze().cpu().numpy()
        heatmap = create_colored_heatmap(pred, target)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img)
        axes[0].set_title('Image')
        
        axes[1].imshow(pred, cmap='gray')
        axes[1].set_title('Prediction')
        
        axes[2].imshow(target, cmap='gray')
        axes[2].set_title('Target')
        
        axes[3].imshow(heatmap)
        axes[3].set_title('Colored Heatmap')
        
        for ax in axes:
            ax.axis('off')
        
        plt.show()
