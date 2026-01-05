import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os

# Let's get everything set up first
np.random.seed(42)
filepath = 'iris.csv'

# We'll borrow the famous Iris dataset from sklearn
print("Fetching the Iris dataset...")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[t] for t in iris.target]

# Real world data is rarely perfect, so let's mess this up a bit!
# I'm going to sprinkle some missing values (NaNs) randomly
print("Sprinkling some missing values...")
for _ in range(5):
    random_row = np.random.randint(0, len(df))
    random_col = np.random.randint(0, 4)
    df.iloc[random_row, random_col] = np.nan

# And let's copy-paste a few rows to create duplicates
print("Creating some duplicates...")
dupes = df.sample(n=3)
df = pd.concat([df, dupes], ignore_index=True)

# Save our "messy" data to a file
df.to_csv(filepath, index=False)
print(f"Done! Saved the data to '{filepath}'.")
print(f"Final shape of the data: {df.shape}")
