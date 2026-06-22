import pandas as pd
import numpy as np
import importlib
from src.data.cleaner import clean_target, clean_numerical_features
from src.features.build_features import add_spatial_features, add_derived_features, encode_categorical_features

def prepare_data(df):
    target_col = 'price'
    
    # 1. Target bereinigen
    df = clean_target(df, target_col)
    
    # 2. Spaltenauswahl definieren
    num_cols = [
        'accommodates', 'bedrooms', 'beds', 'bathrooms', 
        'review_scores_rating', 'number_of_reviews', 'reviews_per_month',
        'minimum_nights', 'maximum_nights',
        'calculated_host_listings_count'
    ]
    
    cat_cols = ['room_type', 'property_type']
    spatial_cols = ['latitude', 'longitude']
    
    # Prüfen, welche Spalten existieren
    valid_num_cols = [c for c in num_cols if c in df.columns]
    valid_cat_cols = [c for c in cat_cols if c in df.columns]
    valid_spatial_cols = [c for c in spatial_cols if c in df.columns]
    
    print(f"Gefundene Numerische Spalten: {valid_num_cols}")
    print(f"Gefundene Kategoriale Spalten: {valid_cat_cols}")
    print(f"Gefundene Spatial Spalten: {valid_spatial_cols}")
    
    # 3. Feature Engineering (Neue Spalten erstellen)
    # A. Spatial
    df, new_spatial_cols = add_spatial_features(df)
    valid_num_cols.extend(new_spatial_cols)
    
    # B. Derived
    df, new_derived_cols = add_derived_features(df)
    valid_num_cols.extend(new_derived_cols)
    
    # 4. Data Cleaning & Transformation
    # A. Numerische Daten auffüllen
    df = clean_numerical_features(df, valid_num_cols)
    
    # B. Kategoriale Daten encoden
    df, dummy_cols_list = encode_categorical_features(df, valid_cat_cols)
    
    # 5. Zusammenführen zu X und y
    final_feature_cols = valid_num_cols + dummy_cols_list
    
    X_raw = df[final_feature_cols].values
    y = df[target_col].values
    
    print(f"\nFinal Shape X: {X_raw.shape}")
    print(f"Final Shape y: {y.shape}")
    
    return X_raw, y

# --- Beispielaufruf (wenn du das Skript direkt ausführst) ---
if __name__ == "__main__":
    # Dummy Data zum Testen
    data = {
        'price': ['$100', '$200', '$50'], 
        'accommodates': [2, 4, 1],
        'latitude': [52.5, 52.6, 52.4],
        'longitude': [13.4, 13.5, 13.3],
        'room_type': ['Entire home', 'Private room', 'Entire home']
    }
    df_test = pd.DataFrame(data)
    
    X, y = prepare_data(df_test)
    print("Test erfolgreich!")

def load_processed_data(path_X='X_filtered.npy', path_y='y.npy'):
    """
    Lägt die gespeicherten Numpy-Dateien und stellt sicher, dass sie Float32 sind.
    """
    try:
        X_raw = np.load(path_X, allow_pickle=True)
        y = np.load(path_y, allow_pickle=True)
        
        # Typ-Korrektur
        X_raw = X_raw.astype(np.float32)
        y = y.astype(np.float32)
        
        print(f"Daten geladen: X Shape {X_raw.shape}, y Shape {y.shape}")
        return X_raw, y
        
    except FileNotFoundError:
        print("Fehler: Dateien nicht gefunden. Hast du prepare_data() ausgeführt?")
        return None, None
    except ValueError as e:
        print(f"Fehler beim Umwandeln in Float: {e}")
        return None, None