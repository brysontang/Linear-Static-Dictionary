import torch
import torch.optim as optim
from util.model import device

def train_memory(model, memory, dataloader, criterion, lr=10, num_epochs=1000):
    memory = memory.to(device)  # Move memory to GPU
    memory.requires_grad_()  # Enable gradient tracking
    optimizer = torch.optim.SGD([memory], lr=lr)

    for param in model.parameters():
        param.requires_grad = False

    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, memory)
            loss = criterion(outputs, targets.argmax(dim=1))
            loss.backward()
            optimizer.step()
            memory.grad.zero_()

            loss_history.append(loss.item())

            # Optional: Clamp memory values to keep them within a desired range
            with torch.no_grad():
                memory.clamp_(-1, 1)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return loss_history

def train_model(model, memory, dataloader, criterion, lr=10, num_epochs=100):
    optimizer = optim.Adam(model.parameters())

    for param in model.parameters():
        param.requires_grad = True

    memory.requires_grad_(False)  # Disable gradient tracking
    loss_history = []

    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, memory)
            loss = criterion(outputs, targets.argmax(dim=1))
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return loss_history