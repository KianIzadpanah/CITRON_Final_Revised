"""
Siamese overlap prediction model.

Architectures supported:
    resnet50          — paper baseline (ResNet-50 encoder, UNet decoder)
    mobilenet_v3_large — lightweight alternative (Reviewer 2 request)

Both share:
    - shared encoder weights (Siamese)
    - concatenated feature maps from both branches
    - UNet-style decoder
    - sigmoid output for pixel-wise overlap probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


# ---------------------------------------------------------------------------
# ResNet-50 Siamese model (paper baseline)
# ---------------------------------------------------------------------------

class SiameseResNet50(nn.Module):
    """
    Shared ResNet-50 encoder (pretrained ImageNet).
    Input: two 256×256 RGB images.
    Feature concat: 2048 + 2048 = 4096 at 8×8 spatial.
    Decoder upsamples to 256×256 sigmoid mask.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Encoder: all layers except avgpool and fc
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # -> (B, 2048, 8, 8) for 256x256 input

        # Decoder: 4096 -> ... -> 1 at 256x256
        dec_channels = [512, 256, 128, 64, 32]
        layers = [DecoderBlock(4096, dec_channels[0])]
        for i in range(1, len(dec_channels)):
            layers.append(DecoderBlock(dec_channels[i - 1], dec_channels[i]))
        self.decoder = nn.Sequential(*layers)
        self.head = nn.Conv2d(dec_channels[-1], 1, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        f1 = self.encode(img1)
        f2 = self.encode(img2)
        x = torch.cat([f1, f2], dim=1)  # (B, 4096, 8, 8)
        x = self.decoder(x)
        x = self.head(x)
        return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# MobileNetV3-Large Siamese model (lightweight alternative)
# ---------------------------------------------------------------------------

class SiameseMobileNetV3(nn.Module):
    """
    Shared MobileNetV3-Large encoder (pretrained ImageNet).
    Last feature map has 960 channels. After concat: 1920.
    Decoder upsamples to 256×256 sigmoid mask.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # features produces (B, 960, 8, 8) for 256x256 input
        self.encoder = backbone.features

        dec_channels = [256, 128, 64, 32, 16]
        layers = [DecoderBlock(1920, dec_channels[0])]
        for i in range(1, len(dec_channels)):
            layers.append(DecoderBlock(dec_channels[i - 1], dec_channels[i]))
        self.decoder = nn.Sequential(*layers)
        self.head = nn.Conv2d(dec_channels[-1], 1, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        f1 = self.encode(img1)
        f2 = self.encode(img2)
        x = torch.cat([f1, f2], dim=1)
        x = self.decoder(x)
        x = self.head(x)
        return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_overlap_model(model_name: str, pretrained: bool = True) -> nn.Module:
    if model_name == "resnet50":
        return SiameseResNet50(pretrained=pretrained)
    elif model_name == "mobilenet_v3_large":
        return SiameseMobileNetV3(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown overlap model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
