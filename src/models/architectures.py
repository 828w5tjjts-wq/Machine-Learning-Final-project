import torch
import torch.nn as nn

class LinearRegression(torch.nn.Module):
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
    
class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPModel, self).__init__()
        
        # Layer 1: Input -> Hidden (64 Neuronen)
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        
        # Layer 2: Hidden -> Hidden (32 Neuronen)
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        
        # Layer 3: Hidden -> Output (1 Neuron)
        self.layer3 = nn.Linear(32, 1)
        
        # Dropout (optional): Schaltet während des Trainings zufällig Neuronen aus,
        # um Overfitting zu verhindern. 20% der Neuronen werden ausgeschaltet.
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Daten durch Layer 1
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        
        # Daten durch Layer 2
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        
        # Daten durch Output Layer (Keine ReLU am Ende, da wir jeden Preis vorhersagen wollen!)
        x = self.layer3(x)
        return x
    
class MLPModel_V2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPModel_V2, self).__init__()
        
        # Layer 1: 562 -> 256 (Mehr Kapazität)
        self.layer1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        
        # Layer 2: 256 -> 128
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        
        # Layer 3: 128 -> 64
        self.layer3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        
        # Layer 4: Output
        self.layer4 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3) # Etwas mehr Dropout

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.relu3(x)
        # Kein Dropout vor dem letzten Layer oft besser
        
        x = self.layer4(x)
        return x
    
class MLPModel_V3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPModel_V3, self).__init__()
        
        # Layer 1: 562 -> 256 (Mehr Kapazität)
        self.layer1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        
        # Layer 2: 256 -> 128
        self.layer2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        
        # Layer 3: 128 -> 64
        self.layer3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        
        # Layer 4: Output
        self.layer4 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.3) # Etwas mehr Dropout

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.relu3(x)
        # Kein Dropout vor dem letzten Layer oft besser
        
        x = self.layer4(x)
        return x
    
class WideMLPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WideMLPModel, self).__init__()
        
        # Layer 1: Input -> Großer Hidden Layer (1024 statt 64)
        self.layer1 = nn.Linear(input_dim, 1024)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1024) # BatchNorm hilft bei großen Netzen
        
        # Layer 2: 1024 -> 512
        self.layer2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(512)
        
        # Layer 3: 512 -> 256
        self.layer3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(256)
        
        # Layer 4: Output
        self.layer4 = nn.Linear(256, 1)
        
        # Dropout (etwas höher, da das Netz größer ist)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        return x