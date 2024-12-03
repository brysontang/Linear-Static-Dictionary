import matplotlib.pyplot as plt
import numpy as np
from util.data_preprocessing import pad_data_to_30x30
from util.model import predict

def visualize_grid(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle("Visualization of 2D Padded Data", fontsize=16)
    
    # Create a custom colormap
    cmap = plt.cm.get_cmap('tab10')
    colors = [cmap(i) for i in range(10)]
    custom_cmap = plt.cm.colors.ListedColormap(['lightgrey'] + colors + ['darkred'])
    
    # Create bounds and norm for the colormap
    bounds = list(range(-2, 11))  # This creates [-2, -1, 0, 1, ..., 10]
    norm = plt.cm.colors.BoundaryNorm(bounds, custom_cmap.N)
    
    im = ax.imshow(data, cmap=custom_cmap, norm=norm, interpolation='nearest')
    ax.set_title("Padded Grid")
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, 30, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 30, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks(range(-1, 11))
    cbar.set_ticklabels(['-1 (pad)'] + list(range(10)) + ['10+'])
    
    plt.tight_layout()
    plt.show()


def visualize_loss(loss_history):
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def visualize_memory(memory):
    # Move to CPU for visualization
    memory_cpu = memory.detach().cpu().numpy()
    plt.imshow(memory_cpu, cmap='gray')
    plt.colorbar()
    plt.show()

def visualize_prediction(input_data, model, memory):
    predicted_output = predict(input_data, model, memory)
    visualize_grid(predicted_output)

def visualize_actual(data):
    visualize_grid(pad_data_to_30x30(data['output']))
    visualize_grid(pad_data_to_30x30(data['input']))
