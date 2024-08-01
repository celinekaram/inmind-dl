import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolution layers
        self.feature = nn.Sequential(
            # Conv 1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0), # 28*28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            
            # Conv 2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
            
            # Conv 3
            nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5, stride = 1), 
            nn.Tanh(),   
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Apply convolutional layers then fully connected layers
        return self.classifier(self.feature(x))

# Instantiate the LeNet5 model
model = LeNet5()
print(model)
summary(model, (1, 32, 32))

# Generate random input tensor of shape (batch_size, channels, height, width)
batch_size = 1
channels = 1
height = 32
width = 32
random_input = torch.rand(batch_size, channels, height, width)

# Print the input
print("Random Input Tensor:")
print(random_input)

# Feed the input through the model
with torch.no_grad():
    output = model(random_input)

# Print the output
print("Output Tensor:")
print(output)