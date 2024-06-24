import torch
import torch.nn.functional as F
from models.model import UNet
import matplotlib.pyplot as plt
from gradcam_define import visualize_gradcam

def run():
    num_class = 6  # Set to the number of classes in combined_mask
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = UNet(num_class).to(device)
    model.load_state_dict(torch.load('path/to/your/model.pth'))  # Laden des trainierten Modells
    model.eval()

    # Beispielbild (ersetzen Sie dies durch den Pfad zu Ihrem Bild)
    img_path = 'path/to/your/image.jpg'
    img = plt.imread(img_path)

    # Grad-CAM Visualisierung
    target_layer = model.e52  # Anpassen an die gewünschte Schicht Ihres Modells
    visualize_gradcam(model, img, target_layer)

# if __name__ == "__main__":
#     run()