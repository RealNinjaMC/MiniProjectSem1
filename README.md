# Data Analysis Mini-Project

Welcome! This project explores two machine learning concepts using Python:
1. **Classification**: Identifying Iris flower species.
2. **Regression**: Predicting salary based on years of experience.

The code is written to be simple and easy to understand.

## Setup

First, let's install the tools we need:

```bash
pip install -r requirements.txt
```

---

## Part 1: Iris Classification

This part classifies flowers into species based on their measurements.

1. **Create Data**:
   ```bash
   python generate_data.py
   ```
   *Creates `iris.csv` with some noise for practice.*

2. **Run Analysis**:
   ```bash
   python analysis.py
   ```
   *Loads data, cleans it, and trains a Logistic Regression model.*

---

## Part 2: Salary Regression

This part predicts how much someone should earn based on their experience.

1. **Create Data**:
   ```bash
   python generate_salary_data.py
   ```
   *Creates `Salary_Data.csv` with some random "messy" data.*

2. **Run Analysis**:
   ```bash
   python regression_analysis.py
   ```
   *Loads data, fills missing values, and trains a Linear Regression model.*

---

## Output

After running the scripts, look in the `output/` folder. You will find:
- **visualizations**: Scatter plots, heatmaps, and confusion matrices.
- **results**: Graphs showing how well the models performed.

Enjoy coding!
