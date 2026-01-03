import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# make sure output folder exists
if not os.path.exists('output'):
    os.makedirs('output')

print("Loading data...")
df = pd.read_csv('iris.csv')
print(df.head())

# cleaning up data
print("\nMissing values before clean:")
print(df.isnull().sum())

# fill numerical nans with mean
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# remove duplicates
print(f"Rows before drop duplicates: {len(df)}")
df.drop_duplicates(inplace=True)
print(f"Rows after: {len(df)}")

# basic eda
print("\nDataset info:")
print(df.info())
print("\nDescription:")
print(df.describe())

# visualization
print("\nGenerating plots...")

# pairplot
sns.pairplot(df, hue='species')
plt.savefig('output/pairplot.png')
plt.close()

# correlation
plt.figure(figsize=(8, 6))
# only use numeric columns for corr
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation')
plt.savefig('output/correlation_heatmap.png')
plt.close()

# boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, orient='h')
plt.title('Distributions')
plt.savefig('output/boxplots.png')
plt.close()

print("Saved plots to output folder")

# modeling
print("\nTraining model...")
X = df.drop('species', axis=1)
y = df['species']

# split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# evaluate
predictions = clf.predict(X_test)
acc = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# confusion matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('output/confusion_matrix.png')
plt.close()

print("done!")
