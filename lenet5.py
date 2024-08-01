import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            #1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            
            #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.classifier(self.feature(x))
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)  # Corrected kernel_size
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  # Corrected kernel_size
        self.pool2 = nn.MaxPool2d(kernel_size= 2, stride=2)

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)  # Corrected kernel_size

        # Fully Connected Layers
        self.fc1 = nn.Linear(self._to_linear, 84)  # Calculated based on conv output
        self.fc2 = nn.Linear(84, 10)  # 10 output classes

    def convs(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        x = F.tanh(self.conv3(x))
        if self._to_linear is None:
            self._to_linear = x.numel()
        return x

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.tanh(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)  # Softmax activation along the class dimension
        return x

model = LeNet5()
print(model)

from torchsummary import summary
summary(model, (1, 32, 32))
# Instantiate the LeNet5 model
model = LeNet5()

# Generate random input tensor of shape (batch_size, channels, height, width)
batch_size = 1
channels = 1
height = 32
width = 32
random_input = torch.rand(batch_size, channels, height, width)

# Feed the input through the model
with torch.no_grad():
    output = model(random_input)

# Print the output
print("Output Tensor:")
print(output)