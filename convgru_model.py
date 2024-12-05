from convgru import ConvGRU
import torch

# Generate a ConvGRU with 3 cells
# input_size and hidden_sizes reflect feature map depths.
# Height and Width are preserved by zero padding within the module.
model = ConvGRU(input_size=8, hidden_sizes=[32,64,16],
                  kernel_sizes=[3, 5, 3], n_layers=3)

x = torch.FloatTensor(1,8,64,64)
output = model(x)

# output is a list of sequential hidden representation tensors
print(type(output)) # list

# final output size
print(output[-1].size()) # torch.Size([1, 16, 64, 64])

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Model parameters
    input_size = 8
    hidden_sizes = [32, 64, 16]
    kernel_sizes = [3, 5, 3]
    n_layers = 3
    
    # Create model
    model = ConvGRU(input_size=input_size, 
                    hidden_sizes=hidden_sizes,
                    kernel_sizes=kernel_sizes, 
                    n_layers=n_layers)
    
    # Generate sample input (batch_size=2, channels=8, height=64, width=64)
    batch_size = 2
    height = 64
    width = 64
    x = torch.randn(batch_size, input_size, height, width)
    
    # Forward pass
    output = model(x)
    
    # Print information about the output
    print(f"Number of output feature maps: {len(output)}")
    for i, out in enumerate(output):
        print(f"Layer {i} output shape: {out.shape}")
        
if __name__ == "__main__":
    main()