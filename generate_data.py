import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_iris_csv(filepath):
    # Load Iris dataset
    iris = load_iris()
    data = iris.data
    target = iris.target
    columns = iris.feature_names
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    df['species'] = [iris.target_names[t] for t in target]
    
    # Introduce data imperfections for cleaning demonstration
    
    # 1. Missing values (NaN)
    n_missing = 5
    for _ in range(n_missing):
        row_idx = np.random.randint(0, len(df))
        col_idx = np.random.randint(0, 4) # Only numeric columns
        df.iloc[row_idx, col_idx] = np.nan
        
    # 2. Duplicate records
    n_duplicates = 3
    # append random existing rows to the dataframe
    duplicates = df.sample(n=n_duplicates)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Dataset generated at: {filepath}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    filepath = os.path.join(os.getcwd(), 'iris.csv')
    generate_iris_csv(filepath)
