import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

# Set plot style
sns.set_style("whitegrid")

# ==============================================================================
# 1. Setup and Data Loading
# ==============================================================================
print("Loading data...")
# We load the raw Ames dataset directly from the source to match the R package's base data
url = "https://raw.githubusercontent.com/topepo/AmesHousing/master/data/ames_raw.csv"
try:
    ames_data = pd.read_csv(url)
except Exception as e:
    print(f"Error loading data: {e}")
    # Fallback to local file if URL fails (user would need to provide file)
    raise

# ==============================================================================
# 2. Handling Missing Values & Integrity Check
# ==============================================================================
print(f"Original dimensions: {ames_data.shape}")

with sqlite3.connect(":memory:") as conn:
    ames_data.to_sql("ames_raw", conn, index=False, if_exists="replace")

    clean_query = """
    SELECT
        [Sale Price] AS Sale_Price,
        [Year Built] AS Year_Built,
        [Year Remod/Add] AS Year_Remod_Add,
        [Kitchen Qual] AS Kitchen_Qual,
        [Gr Liv Area] AS Gr_Liv_Area,
        [Overall Qual] AS Overall_Qual,
        [Yr Sold] AS Year_Sold,
        CASE
            WHEN [Year Remod/Add] > [Year Built] THEN 'Remodeled'
            ELSE 'Not_Remodeled'
        END AS Renovation_Status,
        ([Yr Sold] - [Year Built]) AS House_Age,
        CASE
            WHEN ([Yr Sold] - [Year Built]) <= 0 THEN 1
            ELSE ([Yr Sold] - [Year Built])
        END AS House_Age_Adj,
        CASE [Kitchen Qual]
            WHEN 'Po' THEN 'Poor'
            WHEN 'Fa' THEN 'Fair'
            WHEN 'TA' THEN 'Typical'
            WHEN 'Gd' THEN 'Good'
            WHEN 'Ex' THEN 'Excellent'
            ELSE NULL
        END AS Kitchen_Qual_Label,
        CASE [Overall Qual]
            WHEN 1 THEN 'Very_Poor'
            WHEN 2 THEN 'Poor'
            WHEN 3 THEN 'Fair'
            WHEN 4 THEN 'Below_Average'
            WHEN 5 THEN 'Average'
            WHEN 6 THEN 'Above_Average'
            WHEN 7 THEN 'Good'
            WHEN 8 THEN 'Very_Good'
            WHEN 9 THEN 'Excellent'
            WHEN 10 THEN 'Very_Excellent'
            ELSE NULL
        END AS Overall_Qual_Label
    FROM ames_raw
    WHERE [Sale Price] IS NOT NULL
      AND [Year Built] IS NOT NULL
      AND [Year Remod/Add] IS NOT NULL
      AND [Kitchen Qual] IS NOT NULL
      AND [Gr Liv Area] IS NOT NULL
      AND [Overall Qual] IS NOT NULL
      AND [Yr Sold] IS NOT NULL
    """
    clean_ames_data = pd.read_sql_query(clean_query, conn)

print(f"Cleaned dimensions: {clean_ames_data.shape}")

# Export cleaned data
clean_ames_data.to_csv("clean_ames_data_for_analysis.csv", index=False)

# ==============================================================================
# 3. Deriving Variables
# ==============================================================================
clean_ames_data['Kitchen_Qual'] = pd.Categorical(
    clean_ames_data['Kitchen_Qual_Label'],
    categories=["Poor", "Fair", "Typical", "Good", "Excellent"],
    ordered=True
)

clean_ames_data['Overall_Qual'] = pd.Categorical(
    clean_ames_data['Overall_Qual_Label'],
    categories=[
        "Very_Poor", "Poor", "Fair", "Below_Average", "Average",
        "Above_Average", "Good", "Very_Good", "Excellent", "Very_Excellent"
    ],
    ordered=True
)

clean_ames_data['Renovation_Status'] = pd.Categorical(
    clean_ames_data['Renovation_Status'],
    categories=["Not_Remodeled", "Remodeled"],
    ordered=True
)

# Box-Cox Transformation
# Fit initial model to find lambda
# Statsmodels requires explicit dummy encoding or formula API. We use formula.
# Note: boxcox implementation in scipy returns the transformed array and lambda.
# We use the lambda found in the R script (approx 0.2) or calculate it.
bc_data, fitted_lambda = stats.boxcox(clean_ames_data['Sale_Price'])
print(f"Box-Cox Lambda: {fitted_lambda:.4f}")

# Apply transformation using the calculated lambda
clean_ames_data['SalePrice_bc'] = (clean_ames_data['Sale_Price']**fitted_lambda - 1) / fitted_lambda

# ==============================================================================
# 4. Summary Tables
# ==============================================================================
print("\n--- Figure 1: Response Variable Summary ---")
print(clean_ames_data['Sale_Price'].describe().round(0))

print("\n--- Numerical Predictors Summary ---")
print(clean_ames_data[['House_Age_Adj', 'Gr_Liv_Area']].describe().round(0))

# ==============================================================================
# 5. Model Fitting
# ==============================================================================
# Create transformation columns for models
clean_ames_data['House_Age_sqrt'] = np.sqrt(clean_ames_data['House_Age'])
clean_ames_data['House_Age_log'] = np.log(np.maximum(clean_ames_data['House_Age'], 1)) # ensure no log(0)

# 1. Original (Raw Polynomial equivalent logic)
# Note: R's poly(x, 2) creates orthogonal polynomials by default. 
# To match 'raw=TRUE' or standard interpretation, we often use x + x^2.
model_orig = smf.ols(
    'SalePrice_bc ~ Renovation_Status + House_Age_Adj + Kitchen_Qual + Overall_Qual + Gr_Liv_Area', 
    data=clean_ames_data
).fit()

# 2. Polynomial Age (Age + Age^2)
model_poly = smf.ols(
    'SalePrice_bc ~ Renovation_Status + House_Age_Adj + I(House_Age_Adj**2) + Kitchen_Qual + Overall_Qual + Gr_Liv_Area', 
    data=clean_ames_data
).fit()

# 3. Sqrt Age
model_sqrt = smf.ols(
    'SalePrice_bc ~ Renovation_Status + House_Age_sqrt + Kitchen_Qual + Overall_Qual + Gr_Liv_Area', 
    data=clean_ames_data
).fit()

# 4. Log Age
model_log = smf.ols(
    'SalePrice_bc ~ Renovation_Status + House_Age_log + Kitchen_Qual + Overall_Qual + Gr_Liv_Area', 
    data=clean_ames_data
).fit()

# Comparison Table
comparison = pd.DataFrame({
    'Model': ['Orig', 'PolyAge', 'SqrtAge', 'LogAge'],
    'AIC': [model_orig.aic, model_poly.aic, model_sqrt.aic, model_log.aic],
    'BIC': [model_orig.bic, model_poly.bic, model_sqrt.bic, model_log.bic],
    'Adj_R2': [model_orig.rsquared_adj, model_poly.rsquared_adj, model_sqrt.rsquared_adj, model_log.rsquared_adj]
})
print("\n--- Predictor-transform Comparison ---")
print(comparison.round(3))

# Select Final Model (PolyAge based on R script logic often favoring it)
final_model = model_poly

print("\n--- Final Model Summary ---")
print(final_model.summary())

# ==============================================================================
# 6. Diagnostic Plots
# ==============================================================================
residuals = final_model.resid
fitted_values = final_model.fittedvalues

# 1. Residuals vs Fitted
plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs. Fitted Values (Transformed)')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.savefig('residuals_vs_fitted.png')
plt.close()

# 2. Q-Q Plot
fig = sm.qqplot(residuals, line='45', fit=True)
plt.title('Normal Q-Q Plot (Transformed)')
plt.savefig('qq_plot.png')
plt.close()

# 3. Residuals vs House Age
plt.figure(figsize=(10, 6))
plt.scatter(clean_ames_data['House_Age'], residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs. House Age')
plt.xlabel('House Age')
plt.ylabel('Residuals')
plt.savefig('residuals_vs_age.png')
plt.close()

# 4. Boxplots for Categorical Vars
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.boxplot(x='Kitchen_Qual', y=residuals, data=clean_ames_data, ax=axes[0])
axes[0].set_title('Residuals vs Kitchen Qual')
sns.boxplot(x='Renovation_Status', y=residuals, data=clean_ames_data, ax=axes[1])
axes[1].set_title('Residuals vs Renovation Status')
sns.boxplot(x='Overall_Qual', y=residuals, data=clean_ames_data, ax=axes[2])
axes[2].set_title('Residuals vs Overall Qual')
axes[2].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('residuals_boxplots.png')
plt.close()

# ==============================================================================
# 7. Advanced Diagnostics (VIF & Influence)
# ==============================================================================
# VIF Calculation
# We need the design matrix (exog) from the model
exog = final_model.model.exog
vif_data = pd.DataFrame()
vif_data["Variable"] = final_model.model.exog_names
vif_data["VIF"] = [variance_inflation_factor(exog, i) for i in range(exog.shape[1])]

print("\n--- Variance Inflation Factors (VIF) ---")
print(vif_data.round(3))

# Influence Measures
influence = final_model.get_influence()
cooks_d = influence.cooks_distance[0]

# Threshold for Cook's D (4/n)
n = len(clean_ames_data)
p = len(final_model.params)
cook_threshold = 4 / (n - p - 1)

high_cooks = np.where(cooks_d > cook_threshold)[0]
print(f"\nNumber of points with high Cook's Distance (> {cook_threshold:.4f}): {len(high_cooks)}")

print("\nAnalysis Complete. Plots saved to current directory.")
