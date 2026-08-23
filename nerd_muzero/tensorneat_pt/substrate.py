import torch

def query_cppn_for_weights(cppn_net, input_coords: torch.Tensor):
    """
    Queries the PyTorch-NEAT CPPN network with spatial coordinates
    to generate the weights for a deeper Substrate Network.
    
    Args:
        cppn_net: The PyTorch CPPN instance.
        input_coords: Tensor of shape (N, C) where N is number of connections
                      and C is number of spatial dimensions (e.g., x1, y1, x2, y2).
                      
    Returns:
        Tensor representing the output values mapped from the CPPN.
    """
    with torch.no_grad():
        # Unpack the spatial coordinate columns into the CPPN inputs.
        # PyTorch-NEAT generally takes unpacked inputs: net(x1, y1, x2, y2)
        inputs = [input_coords[:, i] for i in range(input_coords.shape[1])]
        outputs = cppn_net(*inputs)
        
    # If the network has multiple outputs, they are stacked/returned together
    if isinstance(outputs, list):
        return torch.stack(outputs, dim=-1)
        
    return outputs
