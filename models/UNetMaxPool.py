import sys
import os

#Den Projektpfad zu sys.path hinzufügen
project_path = os.path.abspath(os.path.dirname(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)
    
import torch
import torch.nn as nn
from torch.nn.functional import relu





        # Conv2d(in_channels, out_channels, kernel_size, padding, dilation, groups, bias, padding_mode, device, dtype)
        
        # in_chanels:
        # out_channels:
        # kernel_size:
        # padding:
        # dilation:
        # groups:
        # bias:
        # padding_mode:
        # device:
        # dtype:



class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        # -------
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)  # Ersetzt pool1

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)  # Ersetzt pool2

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)  # Ersetzt pool3

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2)  # Ersetzt pool4

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder (unverändert)
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
        '''
        Die zweiten Convolutional-Schichten in jedem Encoder-Block (e12, e22, e32, e42), 
        haben jetzt stride=2, was das Downsampling durchführt wird.
        '''
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))

        xe21 = relu(self.e21(xe12))
        xe22 = relu(self.e22(xe21))

        xe31 = relu(self.e31(xe22))
        xe32 = relu(self.e32(xe31))

        xe41 = relu(self.e41(xe32))
        xe42 = relu(self.e42(xe41))

        xe51 = relu(self.e51(xe42))
        xe52 = relu(self.e52(xe51))

        # Decoder (unverändert)
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out