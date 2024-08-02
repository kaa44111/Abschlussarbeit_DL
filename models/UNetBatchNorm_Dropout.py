import torch
import torch.nn as nn
from torch.nn.functional import relu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UNetBatchNorm(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        # -------
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.b11 = nn.BatchNorm2d(64)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.b12 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.b21 = nn.BatchNorm2d(128)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.b22 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.b31 = nn.BatchNorm2d(256)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.b32 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.b41 = nn.BatchNorm2d(512)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.b42 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.b51 = nn.BatchNorm2d(1024)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.b52 = nn.BatchNorm2d(1024)

        # Dropout layers
        self.dropout = nn.Dropout(0.5)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bd11 = nn.BatchNorm2d(512)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bd12 = nn.BatchNorm2d(512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bd21 = nn.BatchNorm2d(256)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bd22 = nn.BatchNorm2d(256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bd31 = nn.BatchNorm2d(128)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bd32 = nn.BatchNorm2d(128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bd41 = nn.BatchNorm2d(64)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bd42 = nn.BatchNorm2d(64)

        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.b11(self.e11(x)))
        xe12 = relu(self.b12(self.e12(xe11)))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.b21(self.e21(xp1)))
        xe22 = relu(self.b22(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.b31(self.e31(xp2)))
        xe32 = relu(self.b32(self.e32(xe31)))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.b41(self.e41(xp3)))
        xe42 = relu(self.b42(self.e42(xe41)))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.b51(self.e51(xp4)))
        xe52 = relu(self.b52(self.e52(xe51)))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.bd11(self.d11(xu11)))
        xd12 = relu(self.bd12(self.d12(xd11)))
        xd12 = self.dropout(xd12)

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.bd21(self.d21(xu22)))
        xd22 = relu(self.bd22(self.d22(xd21)))
        xd22 = self.dropout(xd22)

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.bd31(self.d31(xu33)))
        xd32 = relu(self.bd32(self.d32(xd31)))
        xd32 = self.dropout(xd32)

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.bd41(self.d41(xu44)))
        xd42 = relu(self.bd42(self.d42(xd41)))
        xd42 = self.dropout(xd42)

        out = self.outconv(xd42)

        return out