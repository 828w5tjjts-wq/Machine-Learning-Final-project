import torch
import torch.nn as nn

class LinearRegression2(torch.nn.Module):
    """
    Linear regression model inherits the torch.nn.Module
    which is the base class for all PyTorch modules.
    """
    def __init__(self, input_dim, output_dim):
        """ Initializes internal Module state. """
        super(LinearRegression, self).__init__()
        # TODO: define a linear layer for the model using torch.nn.Linear
        self.linear = torch.nn.Linear(input_dim, output_dim)


    def forward(self, x):
        """ Defines the computation performed at every call. """
        # TODO: flatten the input to a suitable size for the linear layer 
        # Hint: use x.reshape(...) and the input_dim variable
        
        # Deine Zeile:
        # x = x.reshape(x.shape[0], -1)
        
        # Bessere Alternative (Standard in PyTorch):
        # -1 bedeutet automatisch: "Rechne die zweite Dimension aus"
        x = x.view(x.size(0), -1)
        
        # TODO: run the flattened data through the linear layer and return the outputs
        outputs = self.linear(x)
        return outputs