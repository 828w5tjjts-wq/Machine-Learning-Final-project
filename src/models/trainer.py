import numpy as np
import torch
from fastprogress import master_bar, progress_bar

def train_step(dataloader, optimizer, model, loss_fn, device, master_bar):
    """
    Führt einen Trainingsschritt durch.
    Gibt zurück: (Mean Loss, RMSE)
    """
    model.train()
    batch_losses = []
    sum_squared_errors = 0
    total_samples = 0
    
    for inputs, targets in progress_bar(dataloader, parent=master_bar):
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Shapes korrigieren (Batch, 1) -> (Batch)
        if targets.dim() > 1:
            targets = targets.squeeze(1)

        optimizer.zero_grad()
        predictions = model(inputs)
        
        if predictions.dim() > 1:
            predictions = predictions.squeeze(1)

        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        
        batch_losses.append(loss.item())
        
        # RMSE Berechnung vorbereiten
        error = predictions - targets
        sum_squared_errors += torch.sum(error ** 2).item()
        total_samples += targets.size(0)
    
    mean_loss = sum(batch_losses) / len(batch_losses)
    rmse = np.sqrt(sum_squared_errors / total_samples)
    
    return mean_loss, rmse


def validate_step(dataloader, model, loss_fn, device, master_bar):
    """
    Führt einen Validierungsschritt durch.
    Gibt zurück: (Mean Loss, RMSE)
    """
    model.eval()
    epoch_losses = []
    sum_squared_errors = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in progress_bar(dataloader, parent=master_bar):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            if targets.dim() > 1:
                targets = targets.squeeze(1)

            predictions = model(inputs)
            
            if predictions.dim() > 1:
                predictions = predictions.squeeze(1)
            
            loss = loss_fn(predictions, targets)
            
            epoch_losses.append(loss.item())
            
            # RMSE Berechnung vorbereiten
            error = predictions - targets
            sum_squared_errors += torch.sum(error ** 2).item()
            total_samples += targets.size(0)
    
    mean_loss = sum(epoch_losses) / len(epoch_losses)
    rmse = np.sqrt(sum_squared_errors / total_samples)
    
    return mean_loss, rmse


def run_training(model, optimizer, loss_function, device, num_epochs, train_dataloader, val_dataloader):
    """
    Hauptfunktion, die das Training steuert.
    Funktioniert für jedes Modell (Linear, CNN, Multimodal), solange die DataLoader passen.
    """
    
    train_losses = []
    val_losses = []
    train_rmses = [] 
    val_rmses = []
    
    mb = master_bar(range(num_epochs))
    print(f"Starte Training für {num_epochs} Epochen...")
    
    for epoch in mb:
        
        # 1. Trainieren
        train_loss, train_rmse = train_step(
            dataloader=train_dataloader,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_function,
            device=device,
            master_bar=mb
        )
        train_losses.append(train_loss)
        train_rmses.append(train_rmse)
        
        # 2. Validieren
        val_loss, val_rmse = validate_step(
            dataloader=val_dataloader,
            model=model,
            loss_fn=loss_function,
            device=device,
            master_bar=mb
        )
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)
        
        # 3. Ausgabe
        mb.write(f"Epoch {epoch+1}/{num_epochs} | "
                 f"Train Loss (MSE): {train_loss:.4f} | "
                 f"Train RMSE:       {train_rmse:.2f}€ | "
                 f"Val Loss (MSE):   {val_loss:.4f} | "
                 f"Val RMSE:         {val_rmse:.2f}€")
    
    print("Training beendet.")
    
    return (np.array(train_losses), 
            np.array(val_losses),  
            np.array(train_rmses),   
            np.array(val_rmses))