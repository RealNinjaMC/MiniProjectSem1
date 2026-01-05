import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os

# generating random seed
np.random.seed(42)
filepath = 'iris.csv'

# fetching
print("Fetching the Iris dataset...")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[t] for t in iris.target]

# adding random values 
print("Sprinkling some missing values...")
for _ in range(5):
    random_row = np.random.randint(0, len(df))
    random_col = np.random.randint(0, 4)
    df.iloc[random_row, random_col] = np.nan

# create dupes
print("Creating some duplicates...")
dupes = df.sample(n=3)
df = pd.concat([df, dupes], ignore_index=True)

# saving
df.to_csv(filepath, index=False)
print(f"Done! Saved the data to '{filepath}'.")
print(f"Final shape of the data: {df.shape}")
