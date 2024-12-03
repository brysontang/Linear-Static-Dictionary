import torch
import torch.nn as nn
from util.data_preprocessing import encode_grid, pad_data_to_30x30

# Get the device (GPU if available, otherwise CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
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
        super().__init__()
        
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
        
        # Move model to GPU
        self.to(device)

    def forward(self, grid, memory):
        batch_size = grid.size(0)
        
        # Process grid input
        x_grid = self.relu(self.bn1_grid(self.conv1_grid(grid)))
        
        # Process memory input
        memory = memory.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions
        memory = memory.expand(batch_size, -1, -1, -1)  # Expand to batch size
        memory_features = self.relu(self.bn1_memory(self.conv1_memory(memory)))
        memory_features = self.memory_residual(memory_features)
        
        # Concatenate along channel dimension
        x_combined = torch.cat([x_grid, memory_features], dim=1)
        
        # Rest of the forward pass
        x = self.relu(self.bn_combine(self.conv_combine(x_combined)))
        x = self.residual_blocks(x)
        x = self.conv_final(x)
        return x

def predict(input_data, model, memory):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(encode_grid(pad_data_to_30x30(input_data))).unsqueeze(0).to(device)
        output = model(input_tensor, memory)
        return output.cpu().argmax(dim=1).squeeze().numpy() - 1  # Convert back to [-1, 9] range