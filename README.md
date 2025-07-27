# Ames Home Renovation Premium Prediction

This project uses the Ames Housing dataset to quantify the price premium associated with home renovations. The goal is to predict the impact of home renovations on property prices while accounting for other factors like house age, size, and quality ratings.

## Project Overview

- **Objective**: Quantify the renovation premium in home prices for the Ames Housing dataset using regression models.
- **Data Source**: The Ames Housing dataset (De Cock, 2011), available from the `AmesHousing` R package.
- **Methodology**: Perform data cleaning, feature engineering, and multiple regression modeling. The models include transformations such as Box-Cox, log, and polynomial features to address non-linearity and heteroscedasticity in the data.
- **Outcome**: Derive the key drivers of property value, including house age, renovation status, kitchen quality, and overall quality.

## Key Features of the Project

1. **Data Cleaning & Transformation**:
   - Removed missing values for critical variables such as sale price, kitchen quality, and house age.
   - Derived new features like `Renovation_Status` (whether the house was remodeled) and `House_Age` (the age of the house at the time of sale).
   - Applied a Box-Cox transformation to stabilize the variance in the target variable, `Sale_Price`.

2. **Exploratory Data Analysis**:
   - Summary statistics and visualizations for key variables such as house price, house age, living area, and kitchen quality.
   - Used `ggplot2` for plotting and `gtsummary` for summarizing categorical variables.

3. **Modeling**:
   - Fitted multiple linear regression models to predict house prices, including:
     - A basic linear regression model.
     - A model with polynomial terms for house age.
     - A model with log and square root transformations for house age.
   - Evaluated model performance using R², adjusted R², AIC, and BIC.

4. **Model Diagnostics**:
   - Residuals vs. fitted values plot.
   - Q-Q plot to check normality of residuals.
   - Detection of influential data points using Cook’s distance, leverage, and DFBETAs.
   - Variance inflation factor (VIF) to check for multicollinearity.

5. **Final Model Performance**:
   - The final model achieved an R² of approximately 0.82, indicating a good fit.
   - Key drivers of home price variation include house age, renovation status, kitchen quality, and living area size.

## Installation

### Prerequisites

You will need the following libraries installed to run this project:

- **R**: The project was implemented using R.
- **R Packages**:
    - `AmesHousing`: For accessing the Ames Housing dataset.
    - `dplyr`: For data manipulation.
    - `ggplot2`: For data visualization.
    - `gtsummary`: For summarizing categorical data.
    - `tidyr`: For reshaping and tidying data.
    - `patchwork`: For combining multiple plots.
    - `broom`: For tidying model outputs.
    - `MASS`: For Box-Cox transformation and model fitting.

You can install the necessary R packages using the following code:

```r
# Install required packages
install.packages(c("AmesHousing", "dplyr", "ggplot2", "gtsummary", "tidyr", "patchwork", "broom", "MASS"))
