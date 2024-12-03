import torch
from torch.utils.data import DataLoader, TensorDataset
import json
import os
from util.data_preprocessing import encode_grid, pad_data_to_30x30, augment_data
from util.model import device

def read_data(file_name):
  data_path = os.path.join('..', 'data', 'training', file_name)

  with open(data_path, 'r') as f:
    data = json.load(f)

  return data

def prepare_data(file_name):
  data = read_data(file_name)

  augment_data(data)

  # Create tensors directly on GPU
  X = torch.tensor([encode_grid(pad_data_to_30x30(ex['input'])) for ex in data['train']]).to(device)
  Y = torch.tensor([encode_grid(pad_data_to_30x30(ex['output'])) for ex in data['train']]).to(device)

  dataset = TensorDataset(X, Y)
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

  # Verify everything is on GPU
  print(f"Using device: {device}")

  return dataloader
