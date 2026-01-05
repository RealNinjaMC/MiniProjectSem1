import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

print("--- Starting Iris Analysis ---")

# First things first: we need a folder to save our beautiful plots
if not os.path.exists('output'):
    os.makedirs('output')

# 1. Load the Data
# ----------------
print("\n1. Loading the data...")
try:
    df = pd.read_csv('iris.csv')
    print("Got it! Here's a sneak peek:")
    print(df.head())
except FileNotFoundError:
    print("Oops, I couldn't find 'iris.csv'. Did you run 'generate_data.py'?")
    exit()

# 2. Clean the Data
# -----------------
print("\n2. Cleaning up the mess...")

# Let's check for missing values
print("Checking for missing info...")
missing = df.isnull().sum()
print(missing)

# If we find numbers missing, let's fill them with the average
print("Filling missing numbers with the average...")
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Now let's handle those pesky duplicates
initial_rows = len(df)
print(f"Row count before cleaning: {initial_rows}")
df.drop_duplicates(inplace=True)
print(f"Row count after cleaning: {len(df)}")
print(f"Removed {initial_rows - len(df)} duplicates.")

# 3. Explore the Data
# -------------------
print("\n3. Taking a closer look (EDA)...")
print("Here's the summary stats:")
print(df.describe())

print("Drawing some charts...")

# Pairplot: see how features relate to each other
print("- Saving pairplot...")
sns.pairplot(df, hue='species')
plt.savefig('output/pairplot.png')
plt.close()

# Correlation Heatmap: see what's connected
print("- Saving correlation heatmap...")
plt.figure(figsize=(8, 6))
# We only want to correlate the numbers, not the species names
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title('How features allow correlate')
plt.savefig('output/correlation_heatmap.png')
plt.close()

# Boxplots: see the distribution
print("- Saving boxplots...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, orient='h')
plt.title('Distribution of Features')
plt.savefig('output/boxplots.png')
plt.close()

print("All plots saved to the 'output' folder!")

# 4. Train the Model
# ------------------
print("\n4. Training the brain (Model)...")

# Separate features (X) and answers (y)
X = df.drop('species', axis=1)
y = df['species']

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training a Logistic Regression model...")
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# 5. Evaluate
# -----------
print("\n5. Checking how well we did...")
predictions = clf.predict(X_test)
acc = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {acc*100:.2f}%")
print("Here is the detailed report:")
print(classification_report(y_test, predictions))

# Let's visualize the confusion matrix (where we got confused)
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('output/confusion_matrix.png')
plt.close()

print("Confusion matrix saved.")
print("\nAll done! You're a machine learning wizard now. 🧙‍♂️")
