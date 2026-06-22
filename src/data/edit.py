import numpy as np

def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Teilt Daten in Train, Validation und Test Sets auf.
    Gibt Tupel zurück: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Grenzen berechnen
    train_split = int(train_ratio * num_samples)
    val_split = int((train_ratio + val_ratio) * num_samples)

    # Indices aufteilen
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    print(f"Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # Daten slicen
    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test

import numpy as np

def standardize_data(X_train, X_val, X_test):
    """
    Skaliert die Daten basierend auf Mean und Std des Trainingssets.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    
    # Vermeide Division durch Null (falls eine Konstante im Feature ist)
    std[std == 0] = 1.0

    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    print("Scaling abgeschlossen.")
    return X_train_scaled, X_val_scaled, X_test_scaled