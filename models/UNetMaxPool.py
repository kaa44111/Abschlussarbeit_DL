import sys
import os

#Den Projektpfad zu sys.path hinzufügen
project_path = os.path.abspath(os.path.dirname(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)
    
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Die zweiten Convolutional-Schichten in jedem Encoder-Block (e12, e22, e32, e42), 
haben jetzt stride=2, was das Downsampling durchführt wird.

'''

'''
!!!!Strided Convolutions im Encoder reduzieren die räumlichen Dimensionen stärker als die ursprünglichen MaxPool-Operationen
!!!!F.interpolate(out, size=x.shape[2:]) hinzufügen, um die Größe des Ausgangsbildes an die Eingangsgröße anzupassen.

'''

class UNetMaxPool(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)  # Strided Conv

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)  # Strided Conv

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)  # Strided Conv

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2)  # Strided Conv

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.e11(x))
        xe12 = F.relu(self.e12(xe11))
        #print(f"xe12 shape: {xe12.shape}")

        xe21 = F.relu(self.e21(xe12))
        xe22 = F.relu(self.e22(xe21))
        #print(f"xe22 shape: {xe22.shape}")

        xe31 = F.relu(self.e31(xe22))
        xe32 = F.relu(self.e32(xe31))
        #print(f"xe32 shape: {xe32.shape}")

        xe41 = F.relu(self.e41(xe32))
        xe42 = F.relu(self.e42(xe41))
        #print(f"xe42 shape: {xe42.shape}")

        xe51 = F.relu(self.e51(xe42))
        xe52 = F.relu(self.e52(xe51))
        #print(f"xe52 shape: {xe52.shape}")

        # Decoder
        xu1 = self.upconv1(xe52)
        xu1 = F.interpolate(xu1, size=xe42.shape[2:])  # Anpassung der Größe
        #print(f"xu1 shape: {xu1.shape}")
        xu11 = torch.cat([xu1, xe42], dim=1)
        #print(f"xu11 shape: {xu11.shape}")
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu2 = F.interpolate(xu2, size=xe32.shape[2:])  # Anpassung der Größe
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu3 = F.interpolate(xu3, size=xe22.shape[2:])  # Anpassung der Größe
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu4 = F.interpolate(xu4, size=xe12.shape[2:])  # Anpassung der Größe
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.d41(xu44))
        xd42 = F.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)
        out = F.interpolate(out, size=x.shape[2:])  # Anpassung an die Eingangsgröße

        return out

# # Modell initialisieren
# model = UNetMaxPool(n_class=3)

# # Beispiel-Input
# input_tensor = torch.randn(15, 3, 256, 256)  # Batchgröße 15, 3 Kanäle (RGB), 256x256 Bilder

# # Forward-Pass
# output = model(input_tensor)
# print(f"Output shape: {output.shape}")