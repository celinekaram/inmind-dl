import torch
from torch import nn
from torchsummary import summary

# Define a basic convolutional block with a Conv2D, BatchNorm, and LeakyReLU activation
class MiniBlock(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, stride, padding):
        super(MiniBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(num_filters)
        self.leakyReLU = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyReLU(x)
        return x

# Define a max pooling block
class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MaxPool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.maxpool(x)
        return x

# Define a block that repeats two convolutional operations a specified number of times
class RepetitiveBlock(nn.Module):
    def __init__(self, in_channels_1, num_filters_1, kernel_size_1, stride_1, padding_1, 
                 in_channels_2, num_filters_2, kernel_size_2, stride_2, padding_2, frequency):
        super(RepetitiveBlock, self).__init__()
        self.conv1 = MiniBlock(in_channels_1, num_filters_1, kernel_size_1, stride_1, padding_1)
        self.conv2 = MiniBlock(in_channels_2, num_filters_2, kernel_size_2, stride_2, padding_2)
        self.frequency = frequency

    def forward(self, x):
        for _ in range(self.frequency):
            x = self.conv1(x)
            x = self.conv2(x)
        return x

# Define the YOLOv1 architecture
class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        self.features = nn.Sequential(
            # Layer 1
            MiniBlock(3, 64, 7, 2, 3),
            MaxPool(2, 2),
            # Layer 2
            MiniBlock(64, 192, 3, 1, 1),
            MaxPool(2, 2),
            # Layer 3
            MiniBlock(192, 128, 1, 1, 0),
            MiniBlock(128, 256, 3, 1, 1),
            MiniBlock(256, 256, 1, 1, 0),
            MiniBlock(256, 512, 3, 1, 1),
            MaxPool(2, 2),
            # Layer 4
            RepetitiveBlock(512, 256, 1, 1, 0, 
                            256, 512, 3, 1, 1, 4),
            MiniBlock(512, 512, 1, 1, 0),
            MiniBlock(512, 1024, 3, 1, 1),
            MaxPool(2, 2),
            # Layer 5
            RepetitiveBlock(1024, 512, 1, 1, 0, 
                            512, 1024, 3, 1, 1, 2),
            MiniBlock(1024, 1024, 3, 1, 1),
            MiniBlock(1024, 1024, 3, 2, 1),
            # Layer 6
            MiniBlock(1024, 1024, 3, 1, 1),
            MiniBlock(1024, 1024, 3, 1, 1)
        )

        # Define the fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 7 * 7 * 30),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Example usage
model = YOLOv1()
summary(model, (3, 448, 448))

# Create a random input tensor with shape (batch_size, channels, height, width)
image = torch.rand(3, 3, 448, 448)
output = model(image)
print(output.shape)  # Expected shape: (batch_size, 7*7*30)