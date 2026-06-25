import torch
import matplotlib.pyplot as plt
import numpy as np

def test_learning_rates(learning_rates, model_class, input_dim, output_dim, 
                        train_loader, val_loader, device, epochs=20):
    """
    Testet verschiedene Learning Rates und plottet den Validation Loss.
    
    Args:
        learning_rates (list): Liste der zu testenden Lernraten (z.B. [0.1, 0.01, 0.001])
        model_class: Die Klasse des Modells (z.B. LinearRegression)
        input_dim: Anzahl der Input-Features
        output_dim: Anzahl der Output-Features (meist 1)
        train_loader: DataLoader für Training
        val_loader: DataLoader für Validation
        device: 'cuda' oder 'cpu'
        epochs: Wie viele Epochen pro Lernrate trainiert werden soll
    """
    
    val_losses_history = [] # Speichert den finalen Val Loss für jede LR
    
    print(f"Starte Hyperparameter-Test für {len(learning_rates)} Learning Rates...")
    
    for lr in learning_rates:
        print(f"\n--- Teste Learning Rate: {lr} ---")
        
        # 1. Neues, frisches Modell initialisieren (WICHTIG!)
        model = model_class(input_dim=input_dim, output_dim=output_dim).to(device)
        
        # 2. Optimizer mit aktueller Learning Rate erstellen
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        
        # 3. Kurz trainieren
        model.train()
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Shapes korrigieren
                if targets.dim() > 1: targets = targets.squeeze(1)
                
                optimizer.zero_grad()
                predictions = model(inputs)
                if predictions.dim() > 1: predictions = predictions.squeeze(1)
                
                loss = loss_fn(predictions, targets)
                loss.backward()
                optimizer.step()
        
        # 4. Validieren (Loss auf Validation Set berechnen)
        model.eval()
        val_loss_sum = 0
        count = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if targets.dim() > 1: targets = targets.squeeze(1)
                
                predictions = model(inputs)
                if predictions.dim() > 1: predictions = predictions.squeeze(1)
                
                loss = loss_fn(predictions, targets)
                val_loss_sum += loss.item()
                count += 1
        
        final_val_loss = val_loss_sum / count
        val_losses_history.append(final_val_loss)
        
        print(f"Ergebnis LR {lr}: Finaler Val Loss (MSE) = {final_val_loss:.4f}")

    # --- PLOTTEN ---
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, val_losses_history, marker='o', linestyle='-', color='b')
    plt.xscale('log')  # Logarithmische Skala für die X-Achse (Standard für LR)
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Validation Loss (MSE)')
    plt.title('Einfluss der Learning Rate auf den Validation Loss')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Den besten Punkt markieren
    best_idx = np.argmin(val_losses_history)
    best_lr = learning_rates[best_idx]
    best_loss = val_losses_history[best_idx]
    plt.scatter(best_lr, best_loss, color='red', s=100, zorder=5, label=f'Bester: {best_lr}')
    plt.legend()
    
    plt.show()
    
    return best_lr, best_loss


def test_learning_rates_curves(learning_rates, model_class, input_dim, output_dim, 
                               train_loader, val_loader, device, epochs=20):
    """
    Testet verschiedene Learning Rates und plottet den gesamten Verlauf des Validation Loss.
    """
    
    # Dictionary um die Verläufe zu speichern: Key = LR, Value = Liste der Losses
    all_val_losses = {}
    
    print(f"Starte Hyperparameter-Test für {len(learning_rates)} Learning Rates ({epochs} Epochen)...")
    
    for lr in learning_rates:
        print(f"\n--- Teste Learning Rate: {lr} ---")
        
        model = model_class(input_dim=input_dim, output_dim=output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        
        epoch_losses = [] # Speichert Loss für jede Epoche
        
        for epoch in range(epochs):
            # Training
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if targets.dim() > 1: targets = targets.squeeze(1)
                
                optimizer.zero_grad()
                predictions = model(inputs)
                if predictions.dim() > 1: predictions = predictions.squeeze(1)
                
                loss = loss_fn(predictions, targets)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss_sum = 0
            count = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    if targets.dim() > 1: targets = targets.squeeze(1)
                    
                    predictions = model(inputs)
                    if predictions.dim() > 1: predictions = predictions.squeeze(1)
                    
                    val_loss_sum += loss_fn(predictions, targets).item()
                    count += 1
            
            avg_val_loss = val_loss_sum / count
            epoch_losses.append(avg_val_loss)
            
        # Speichern
        all_val_losses[lr] = epoch_losses
        print(f"LR {lr} beendet. Finaler Val Loss: {avg_val_loss:.4f}")

    # --- PLOTTEN ---
    plt.figure(figsize=(12, 6))
    
    for lr, losses in all_val_losses.items():
        plt.plot(range(epochs), losses, label=f'LR: {lr}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss (MSE)')
    plt.title('Vergleich der Learning Rates über die Zeit')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Besten finalen Loss finden (optional)
    best_final_lr = min(all_val_losses.items(), key=lambda x: x[1][-1])[0]
    print(f"\nBester finaler Val Loss bei LR: {best_final_lr}")
    
    return all_val_losses