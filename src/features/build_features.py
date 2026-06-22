import pandas as pd
import numpy as np

def add_spatial_features(df):
    """
    Berechnet die Distanz zum geometrischen Zentrum aller Wohnungen.
    """
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Entferne Zeilen ohne Koordinaten
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Zentrum berechnen
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        # Euklidische Distanz berechnen
        df['dist_to_center'] = np.sqrt((df['latitude'] - center_lat)**2 + (df['longitude'] - center_lon)**2)
        
        return df, ['dist_to_center'] # Rückgabe: DataFrame und Liste der neuen Spalten
    return df, []

def add_derived_features(df):
    """
    Erstellt abgeleitete Features wie Preis pro Person.
    """
    new_cols = []
    
    # Preis pro Person
    if 'accommodates' in df.columns:
        # Vermeide Division durch Null
        df['price_per_person'] = df['price'] / df['accommodates'].replace(0, 1)
        new_cols.append('price_per_person')
        
    return df, new_cols

def encode_categorical_features(df, cat_cols):
    """
    Führt One-Hot Encoding durch und gibt die neuen Dummy-Spalten zurück.
    """
    dummy_cols_list = []
    for col in cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        dummy_cols_list.extend(list(dummies.columns))
    
    return df, dummy_cols_list