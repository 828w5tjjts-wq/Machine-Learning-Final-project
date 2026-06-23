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