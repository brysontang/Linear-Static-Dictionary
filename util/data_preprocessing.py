import numpy as np

def pad_data_to_30x30(data):
    # Convert input to numpy array if it's not already
    data = np.array(data)
    
    # Initialize a 30x30 array filled with -1
    padded_data = np.full((30, 30), -1, dtype=int)
    
    # Copy data to padded_data
    rows, cols = data.shape
    padded_data[:rows, :cols] = data[:30, :30]
    
    return padded_data

def augment_data(data):        
    # Apply Caesar cipher to non-zero/non-negative-one values
    def caesar_cipher(grid, step):
        result = []
        for row in grid:
            new_row = []
            for val in row:
                if val not in [0, -1]:
                    # Apply cipher in range 1-9
                    new_val = ((val - 1 + step) % 9) + 1
                    new_row.append(new_val)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result

    # Create augmented examples with different cipher shifts
    augmented_examples = []
    for example in data['train']:
        input_grid = example['input']
        output_grid = example['output']
        
        # Try different shifts between 1-9
        for shift in range(1, 10):
            new_input = caesar_cipher(input_grid, shift)
            new_output = caesar_cipher(output_grid, shift)
            augmented_examples.append({
                'input': new_input,
                'output': new_output
            })
            
    # Add augmented examples to training data
    data['train'].extend(augmented_examples)


def encode_grid(grid):
    """Convert a 2D grid to a 11-channel 3D tensor."""
    encoded = np.zeros((11, 30, 30), dtype=np.float32)
    for i in range(30):
        for j in range(30):
            value = grid[i, j]
            if -1 <= value <= 9:
                encoded[value + 1, i, j] = 1
    return encoded