# Ames Home Renovation Premium Prediction

This project uses the Ames Housing dataset to quantify the price premium associated with home renovations. The goal is to predict the impact of home renovations on property prices while accounting for other factors like house age, size, and quality ratings.

## Project Overview

- **Objective**: Quantify the renovation premium in home prices using Python-based regression analysis.
- **Data Source**: The raw Ames Housing dataset (De Cock, 2011), accessed directly from the [AmesHousing GitHub repository](https://github.com/topepo/AmesHousing).
- **Methodology**: Data cleaning, feature engineering, and multiple linear regression (OLS). Models include Box-Cox transformation on the target variable and polynomial features for house age.
- **Outcome**: Key drivers identified include house age, renovation status, kitchen quality, and overall quality.

## Key Features

1. **Data Cleaning & Transformation**:
   - Automated fetching and cleaning of raw data.
   - Derivation of `Renovation_Status` and `House_Age`.
   - Box-Cox transformation of `Sale_Price` to address non-normality.
   
2. **Modeling**:
   - Implementation of multiple OLS models using `statsmodels`:
     - Linear Age
     - Polynomial Age (Age + Age²)
     - Log and Square Root Age transformations
   - Model selection based on AIC/BIC and Adjusted R².

3. **Diagnostics**:
   - Residual analysis (Residuals vs Fitted, Q-Q Plots).
   - Multicollinearity check using VIF (Variance Inflation Factor).
   - Influence detection using Cook's Distance.

## Installation & Usage

### Prerequisites

You need **Python 3.7+** and the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `scipy`

### Setup

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn statsmodels scipy

### Running the Analysis

Run the main analysis script:

```bash
python ames_analysis.py

```

This will:

1. Fetch the dataset.
2. Perform the regression analysis.
3. Output summary tables to the console.
4. Save diagnostic plots (e.g., `residuals_vs_fitted.png`, `qq_plot.png`) to the current directory.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
