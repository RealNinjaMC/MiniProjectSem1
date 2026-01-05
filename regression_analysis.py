import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

print("Starting salary analysis")

# Create output folder if it doesn’t exist
if not os.path.exists('output'):
    os.makedirs('output')


# 1. Load the data
print("\nLoading the dataset")
try:
    df = pd.read_csv('Salary_Data.csv')
    print("Dataset loaded successfully")
    print(df.head())
except FileNotFoundError:
    print("Salary_Data.csv not found")
    print("Run the data generation file first")
    exit()


# 2. Clean the data
print("\nCleaning the data")

# Handle missing values (safe & future-proof)
missing = df.isnull().sum()
if missing.sum() > 0:
    print("Missing values found, filling with mean")
    df['YearsExperience'] = df['YearsExperience'].fillna(df['YearsExperience'].mean())
    df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
else:
    print("No missing values found")

# Remove duplicates
before = len(df)
df = df.drop_duplicates()
after = len(df)

if before != after:
    print(f"Removed {before - after} duplicate rows")
else:
    print("No duplicate rows found")


# 3. Explore the data
print("\nExploring the data")
print(df.describe())

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='YearsExperience', y='Salary', data=df, s=100)
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.grid(True, alpha=0.3)
plt.savefig('output/salary_scatter.png')
print("Scatter plot saved in output folder")
plt.close()


# 4. Train the model
print("\nTraining the model")

X = df[['YearsExperience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")

model = LinearRegression()
model.fit(X_train, y_train)

print("Model training completed")
print(f"Base salary: ${model.intercept_:.2f}")
print(f"Increase per year of experience: ${model.coef_[0]:.2f}")


# 5. Test the model
print("\nTesting the model")

predictions = model.predict(X_test)
score = r2_score(y_test, predictions)

print(f"R2 score: {score:.4f}")

# Regression result plot
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Actual Data')
plt.plot(X_test, predictions, linewidth=2, label='Predicted Line')
plt.title('Salary Prediction Result')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('output/salary_regression_result.png')
print("Regression result saved in output folder")
plt.close()


# 6. Make a prediction (no sklearn warning)
print("\nMaking a sample prediction")

years = 6.5
years_df = pd.DataFrame([[years]], columns=['YearsExperience'])
pred_salary = model.predict(years_df)[0]

print(f"Estimated salary for {years} years of experience: ${pred_salary:,.2f}")

print("\nProcess completed successfully")
