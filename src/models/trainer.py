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

def train_model_with_early_stopping(model, train_dataloader, val_dataloader, loss_function, optimizer, num_epochs, device, patience=5, tolerance=1e-4):
    """
    Trainingsschleife mit Early Stopping Logik und RMSE Berechnung.
    Stoppt das Training, wenn sich der Validation Loss für 'patience' Epochen nicht verbessert.
    """
    
    train_losses = []
    val_losses = []
    train_rmses = []
    val_rmses = []
    
    # Track best validation loss and epochs without improvement
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"Starte Training mit Early Stopping (Patience={patience})...")
    
    # Iterate over epochs
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        sum_squared_errors = 0
        total_samples = 0
        
        # --- TRAINING ---
        for inputs, targets in train_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Shapes korrigieren (Regression)
            if targets.dim() > 1:
                targets = targets.squeeze(1)
            
            # Forward pass
            outputs = model(inputs)
            if outputs.dim() > 1:
                outputs = outputs.squeeze(1)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Zero gradients, Backward, Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            
            # RMSE Berechnung vorbereiten
            error = outputs - targets
            sum_squared_errors += torch.sum(error ** 2).item()
            total_samples += targets.size(0)
        
        # Calculate average training loss for epoch
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Train RMSE berechnen
        train_rmse = np.sqrt(sum_squared_errors / total_samples)
        train_rmses.append(train_rmse)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        val_sum_squared_errors = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for val_inputs, val_targets in val_dataloader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                
                if val_targets.dim() > 1:
                    val_targets = val_targets.squeeze(1)
                
                val_outputs = model(val_inputs)
                if val_outputs.dim() > 1:
                    val_outputs = val_outputs.squeeze(1)
                
                val_loss += loss_function(val_outputs, val_targets).item()
                
                # RMSE Berechnung vorbereiten
                error = val_outputs - val_targets
                val_sum_squared_errors += torch.sum(error ** 2).item()
                val_total_samples += val_targets.size(0)
        
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
        # Val RMSE berechnen
        val_rmse = np.sqrt(val_sum_squared_errors / val_total_samples)
        val_rmses.append(val_rmse)
        
        # --- CHECK IMPROVEMENT ---
        # Wir nutzen den Val Loss für die Early Stopping Entscheidung (Standard)
        if val_loss < best_val_loss - tolerance:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Print loss values
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train RMSE: {train_rmse:.2f}€ | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.2f}€")
        
        # --- EARLY STOPPING CHECK ---
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    return (np.array(train_losses), 
            np.array(val_losses),  
            np.array(train_rmses),   
            np.array(val_rmses))