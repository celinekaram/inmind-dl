import torch
import torch.nn as nn
import torch.nn.functional as F

# Inspiration:
# - UNET: encoder - bottleneck - decoder - skip connections
# - DeepLab: atrous convolution used to capture wider context without increasing the number of parameters

def conv_block(in_channels, out_channels, dilation=1):
    """
    A simple convolutional block consisting of: Convolutional Layer, Batch Normalization, ReLU Activation
    This block can include dilation to capture wider context (inspired by atrous convolution in DeepLab).
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class CustomModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomModel, self).__init__()

        # Encoder: Downsampling path with increasing feature maps
        self.enc1 = conv_block(in_channels, 64)  # Layer 1
        self.enc2 = conv_block(64, 128, dilation=2)  # Layer 2 with dilation
        self.enc3 = conv_block(128, 256, dilation=4)  # Layer 3 with more dilation

        # Bottleneck: Capture the highest level features with atrous convolution
        self.bottleneck = nn.Conv2d(256, 512, kernel_size=3, padding=4, dilation=4)

        # Decoder: Upsampling path with skip connections
        self.dec3 = conv_block(256 + 512, 256)  # Layer 3
        self.dec2 = conv_block(128 + 256, 128)  # Layer 2
        self.dec1 = conv_block(64 + 128, 64)    # Layer 1

        # Final classification layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1(x)  # First encoding layer
        enc2 = self.enc2(F.max_pool2d(enc1, 2))  # Pool and pass to next
        enc3 = self.enc3(F.max_pool2d(enc2, 2))  # Same here

        # Bottleneck with atrous convolution for capturing a wide receptive field
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))

        # Decoder with skip connections (concatenation)
        dec3 = torch.cat([F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True), enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = torch.cat([F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = torch.cat([F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Final classification layer
        return self.final(dec1)

def main():
    # Instantiate and test the model
    model = CustomModel(in_channels=3, num_classes=10)
    output = model(torch.randn(1, 3, 1280, 720))  # Input: Batch of 1, 3 channels (RGB), 1280x720
    print(output.shape)  # Should output (1, 10, 1280, 720) - 10 channels for 10 classes

if __name__ == "__main__":
    main()