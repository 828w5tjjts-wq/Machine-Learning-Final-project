import pandas as pd
import numpy as np

def clean_target(df, target_col='price'):
    """
    Bereinigt die Zielvariable (Price).
    """
    # Preis-Säule bereinigen
    df[target_col] = df[target_col].replace(r'[\$,]', '', regex=True).astype(float)
    
    # Entferne ungültige Preise (NaNs und extreme Ausreißer)
    df = df.dropna(subset=[target_col])
    df = df[(df[target_col] > 10) & (df[target_col] < 2000)]
    
    return df

def clean_numerical_features(df, num_cols):
    """
    Füllt fehlende Werte in numerischen Spalten mit dem Median.
    """
    for col in num_cols:
        median_val = df[col].median()
        if pd.isna(median_val): 
            median_val = 0
        df[col] = df[col].fillna(median_val)
    return df

