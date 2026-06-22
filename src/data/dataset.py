import torch
from torch.utils.data import Dataset, DataLoader

class AirbnbDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # Shape (N,) -> (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64):
    """
    Erstellt PyTorch DataLoader für Training, Validation und Test.
    """
    train_dataset = AirbnbDataset(X_train, y_train)
    val_dataset = AirbnbDataset(X_val, y_val)
    test_dataset = AirbnbDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"DataLoader bereit (Batch Size {batch_size}):")
    print(f"  Train Batches: {len(train_loader)}")
    print(f"  Val Batches: {len(val_loader)}")
    print(f"  Test Batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader