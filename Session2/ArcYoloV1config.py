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

# Define the architecture configuration
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# Define a max pooling block
class MaxPool(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(MaxPool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.maxpool(x)

# Define the YOLOv1 architecture
class YOLOv1(nn.Module):
    def __init__(self, input_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = input_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers.append(MiniBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]))
                in_channels = x[1]
            elif x == "M":
                layers.append(MaxPool())
            elif type(x) == list:
                conv1 = x[0]  # Tuple
                conv2 = x[1]  # Tuple
                repeats = x[2]  # Int

                for i in range(repeats):
                    layers.append(MiniBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]))
                    in_channels = conv1[1]
                    layers.append(MiniBlock(in_channels, conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]))
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (5 * B + C)),  # Final layer with the desired output
            nn.LeakyReLU(0.1)
        )

# Example usage
model = YOLOv1(split_size=7, num_boxes=2, num_classes=20)
summary(model, (3, 448, 448))

# Create a random input tensor with shape (batch_size, channels, height, width)
image = torch.rand(3, 3, 448, 448)
output = model(image)
print(output.shape)  # Expected shape: (batch_size, 1470)