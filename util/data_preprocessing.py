import numpy as np
import matplotlib.pyplot as plt

def pad_data_to_30x30(data):
    # Convert input to numpy array if it's not already
    data = np.array(data)
    
    # Initialize a 30x30 array filled with -1
    padded_data = np.full((30, 30), -1, dtype=int)
    
    # Copy data to padded_data
    rows, cols = data.shape
    padded_data[:rows, :cols] = data[:30, :30]
    
    return padded_data

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


def encode_grid(grid):
    """Convert a 2D grid to a 11-channel 3D tensor."""
    encoded = np.zeros((11, 30, 30), dtype=np.float32)
    for i in range(30):
        for j in range(30):
            value = grid[i, j]
            if -1 <= value <= 9:
                encoded[value + 1, i, j] = 1
    return encoded