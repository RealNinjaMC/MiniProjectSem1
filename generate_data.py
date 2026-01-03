import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os

# setup
np.random.seed(42)
filepath = 'iris.csv'

# load the data from sklearn
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[t] for t in iris.target]

# add some noise to the data
# adding nans
for _ in range(5):
    r = np.random.randint(0, len(df))
    c = np.random.randint(0, 4)
    df.iloc[r, c] = np.nan

# adding duplicates
dupes = df.sample(n=3)
df = pd.concat([df, dupes], ignore_index=True)

# save it
df.to_csv(filepath, index=False)
print(f"Saved data to {filepath}, shape: {df.shape}")
