import torch
import torch.nn as nn
from torchsummary import summary

# Custom Module Subclass
class my_NN(nn.Module):
    def __init__(self):
        super(my_NN, self).__init__()
        
        # Define the layers
        self.layer1 = nn.Linear(4, 6)
        self.layer2 = nn.Linear(6, 8)
        self.layer3 = nn.Linear(8, 4)
        self.layer4 = nn.Linear(4, 4)
        self.layer5 = nn.Linear(4, 2)
        
        # Define the activation functions
        
        # *Leaky ReLU*: similar to ReLU, returns a small negative value instead of 0 for negative inputs (unit is not active)
        self.leaky_relu = nn.LeakyReLU()
        
        # *tanh*: -1 < output < 1, normalized to have mean = 0
        self.tanh = nn.Tanh()
        
        # *sigmoid*: 0 < output < 1 => probability
        self.sigmoid = nn.Sigmoid()
        
        # *ELU*: Exponential Linear Unit: smooth the gradient to avoid vanishing/exploding gradients
        self.elu = nn.ELU()
        
        # Softmax: Converts the outputs to probability distributions over the output classes (2 outputs: 0 or 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.leaky_relu(self.layer1(x))
        x = self.tanh(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        x = self.elu(self.layer4(x))
        x = self.softmax(self.layer5(x))
        return x

# Instantiate the neural network
model = my_NN()
print(model)

# Define batch and input
batch_nb = 16
input_dim = 4

# Generate a random input tensor
input_tensor = torch.rand(batch_nb, input_dim)

# Feed the input through the neural network
with torch.no_grad(): # 
    output_tensor = model(input_tensor)

# Print the output
print("Input Tensor:")
print(input_tensor)
print("\nOutput Tensor:")
print(output_tensor)

# Use torch summary to print a summary of model
summary(model)