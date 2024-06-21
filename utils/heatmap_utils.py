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
    height, width = target.shape[1], target.shape[2]
    heatmap = np.zeros((height, width, 3), dtype=np.float32)
    
    for class_index in range(pred.shape[0]):
        pred_class = pred[class_index]
        target_class = target[class_index]

        # Debug-Ausgaben
        print(f"Class index: {class_index}")
        print(f"Pred shape: {pred_class.shape}, Target shape: {target_class.shape}")
        
        # True Positives (richtig als Positiv erkannt) in Grün
        heatmap[(pred_class > 0.5) & (target_class > 0.5)] = [0, 1, 0]
        
        # False Negatives (falsch als Negativ erkannt) in Rot
        heatmap[(pred_class <= 0.5) & (target_class > 0.5)] = [1, 0, 0]
        
        # False Positives (falsch als Positiv erkannt) in Blau
        heatmap[(pred_class > 0.5) & (target_class <= 0.5)] = [0, 0, 1]
    
    return heatmap

def visualize_colored_heatmaps(images, preds, targets):
    """Visualisiert die Bilder, Vorhersagen, Zielwerte und Heatmaps."""
    batch_size = images.shape[0]
    
    for i in range(batch_size):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        pred = preds[i].cpu().numpy()
        target = targets[i].cpu().numpy()
        heatmap = create_colored_heatmap(pred, target)

        # Debug-Ausgaben
        print(f"Image shape: {img.shape}, Pred shape: {pred.shape}, Target shape: {target.shape}")
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img)
        axes[0].set_title('Image')
        
        # Zeige die erste Klasse als Beispiel an
        axes[1].imshow(pred[0], cmap='gray')
        axes[1].set_title('Prediction (Class 1)')
        
        axes[2].imshow(target[0], cmap='gray')
        axes[2].set_title('Target (Class 1)')
        
        axes[3].imshow(heatmap)
        axes[3].set_title('Colored Heatmap')
        
        for ax in axes:
            ax.axis('off')
        
        plt.show()
