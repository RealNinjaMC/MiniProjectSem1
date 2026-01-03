# Iris Data Analysis

A simple Python project to explore and classify the Iris dataset.

## Setup

First, grab the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Generate Data**: This script pulls the data from sklearn and adds some noise (missing values, duplicates) just so we have something to clean up.
   ```bash
   python generate_data.py
   ```

2. **Run Analysis**: This does the heavy lifting—cleaning, EDA, plotting, and modeling.
   ```bash
   python analysis.py
   ```

## Output

- **Plots**: Check the `output/` folder for pairplots, heatmaps, and confusion matrices.
- **Model**: Uses Logistic Regression. Currently hitting **100% accuracy** on the test set.

## Files

- `generate_data.py`: Creates `iris.csv`.
- `analysis.py`: Main script for stats and ML.
- `output/`: Generated images.
