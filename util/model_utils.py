import torch
import torch.nn as nn
from util.model import EnhancedMultiChannelCNN, device

def initialize_model():
    # Create model and move to GPU
    model = EnhancedMultiChannelCNN().to(device)
    
    # Initialize criterion
    criterion = nn.CrossEntropyLoss().to(device)
    
    print(f"Model device: {next(model.parameters()).device}")

    return model, criterion

def initialize_memory():
  memory = torch.rand(30, 30, device=device) * 2 - 1

  print(f"Memory tensor device: {memory.device}")
  return memory
