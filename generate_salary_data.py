import pandas as pd
import numpy as np

# Fix the random seed so we get the same output every time
np.random.seed(42)

# CSV file name
filename = 'Salary_Data.csv'

print("Creating sample salary data...")

# Number of employees
num_people = 50


# Generate years of experience (1.1 to 10.5 years)
experience = np.random.uniform(1.1, 10.5, num_people)
experience = np.round(experience, 1)


# Calculate salary based on experience + some randomness
noise = np.random.normal(0, 4000, num_people)
salary = 38000 + (9500 * experience) + noise
salary = np.round(salary, 0)


# Create DataFrame
df = pd.DataFrame({
    'YearsExperience': experience,
    'Salary': salary
})


# Add a few missing values on purpose
for _ in range(3):
    row = np.random.randint(0, len(df))
    col = np.random.randint(0, 2)
    df.iloc[row, col] = np.nan


# Add some duplicate rows
duplicates = df.sample(n=2)
df = pd.concat([df, duplicates], ignore_index=True)


# Save to CSV
df.to_csv(filename, index=False)

print(f"Saved {len(df)} rows to {filename}")
print(df.head())
