import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import v2

"""
!!!!!! Dies ist ein ChatGPT Code!!!!!
    ###Erklärung###
    GradCAM-Klasse: Diese Klasse berechnet die Grad-CAM.
    preprocess_image: Diese Funktion bereitet das Bild für das Modell vor.
    show_cam_on_image: Diese Funktion erstellt die Heatmap mit Matplotlib und überlagert sie auf dem Originalbild.
    visualize_gradcam: Funktion zur Visualisierung der Grad-CAM-Heatmaps.
    Beispielaufruf: Laden eines vortrainierten Modells und eines Bildes, und Visualisieren der Grad-CAM."""

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        target = output[0][class_idx]
        target.backward()

        gradient = self.gradient.cpu().data.numpy()[0]
        activation = self.activation.cpu().data.numpy()[0]

        weights = np.mean(gradient, axis=(1, 2))
        cam = np.zeros(activation.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activation[i]

        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def preprocess_image(img):
    preprocessing = v2.Compose([
        v2.ToPureTensor(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return preprocessing(img).unsqueeze(0)

def show_cam_on_image(img, mask):
    mask = torch.tensor(mask)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(img.shape[1], img.shape[2]), mode='bilinear', align_corners=False)
    mask = mask.squeeze().cpu().numpy()
    heatmap = plt.get_cmap('jet')(mask)[:, :, :3]
    heatmap = torch.tensor(heatmap).permute(2, 0, 1)
    img = torch.tensor(img).permute(2, 0, 1)
    cam = heatmap * 0.4 + img * 0.6
    return cam.permute(1, 2, 0).numpy()

def visualize_gradcam(model, img, target_layer, class_idx=None):
    input_img = preprocess_image(img)

    grad_cam = GradCAM(model, target_layer)
    mask = grad_cam(input_img, class_idx)

    cam_img = show_cam_on_image(img, mask)
    plt.imshow(cam_img)
    plt.show()


