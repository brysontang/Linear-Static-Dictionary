import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from util.data_preprocessing import encode_grid, pad_data_to_30x30, visualize_grid

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class EnhancedMultiChannelCNN(nn.Module):
    def __init__(self, num_residual_blocks=3):
        super(EnhancedMultiChannelCNN, self).__init__()
        
        # Grid processing branch
        self.conv1_grid = nn.Conv2d(11, 32, kernel_size=3, padding=1)
        self.bn1_grid = nn.BatchNorm2d(32)
        
        # Memory processing branch
        self.conv1_memory = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1_memory = nn.BatchNorm2d(32)
        self.memory_residual = ResidualBlock(32)
        
        # Combined processing
        self.conv_combine = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_combine = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        
        self.conv_final = nn.Conv2d(64, 11, kernel_size=3, padding=1)

    def forward(self, grid, memory):
        batch_size = grid.size(0)
        
        # Process grid input
        x_grid = self.relu(self.bn1_grid(self.conv1_grid(grid)))  # [batch_size, 32, 30, 30]
        
        # Process memory input
        memory = memory.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions [1, 1, 30, 30]
        memory = memory.expand(batch_size, -1, -1, -1)  # Expand to batch size [batch_size, 1, 30, 30]
        memory_features = self.relu(self.bn1_memory(self.conv1_memory(memory)))  # [batch_size, 32, 30, 30]
        memory_features = self.memory_residual(memory_features)
        
        # Concatenate along channel dimension
        x_combined = torch.cat([x_grid, memory_features], dim=1)  # [batch_size, 64, 30, 30]
        
        # Rest of the forward pass
        x = self.relu(self.bn_combine(self.conv_combine(x_combined)))
        x = self.residual_blocks(x)
        x = self.conv_final(x)
        return x
    
def train_memory(model, memory, dataloader, criterion, lr=10, num_epochs=1000):
    for param in model.parameters():
        param.requires_grad = False

    memory.requires_grad_()  # Enable gradient tracking
    optimizer = torch.optim.SGD([memory], lr=lr)

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs, memory)
            loss = criterion(outputs, targets.argmax(dim=1))
            loss.backward()
            
            # print('Memory Gradient Norm:', memory_grad_norm)
            
            optimizer.step()
            memory.grad.zero_()

            # Optional: Clamp memory values to keep them within a desired range
            with torch.no_grad():
                memory.clamp_(-1, 1)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def train_model(model, memory, dataloader, criterion, lr=10, num_epochs=100):
    optimizer = optim.Adam(model.parameters())

    for param in model.parameters():
        param.requires_grad = True

    memory.requires_grad_(False)  # Disable gradient tracking

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs, memory)
            loss = criterion(outputs, targets.argmax(dim=1))
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def visualize_memory(memory):
    plt.imshow(memory.detach().numpy(), cmap='gray')
    plt.colorbar()
    plt.show()

def predict(input_data, model, memory):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(encode_grid(pad_data_to_30x30(input_data))).unsqueeze(0)
        output = model(input_tensor, memory)
        return output.argmax(dim=1).squeeze().numpy() - 1  # Convert back to [-1, 9] range
    
# Function to visualize prediction
def visualize_prediction(input_data, model, memory):
    predicted_output = predict(input_data, model, memory)
    visualize_grid(predicted_output)

def visualize_actual(data):
    visualize_grid(pad_data_to_30x30(data['output']))
    visualize_grid(pad_data_to_30x30(data['input']))
