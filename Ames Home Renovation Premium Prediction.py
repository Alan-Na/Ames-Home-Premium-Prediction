import pandas as pd
import numpy as np
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

# Rename columns to match the R script's snake_case conventions
# The R function `make_ames()` renames specific columns; we map the raw names to those.
column_map = {
    'Sale Price': 'Sale_Price',
    'Year Built': 'Year_Built',
    'Year Remod/Add': 'Year_Remod_Add',
    'Kitchen Qual': 'Kitchen_Qual',
    'Gr Liv Area': 'Gr_Liv_Area',
    'Overall Qual': 'Overall_Qual',
    'Yr Sold': 'Year_Sold'
}
ames_data = ames_data.rename(columns=column_map)

# Select variables used in the analysis
selected_vars = ['Sale_Price', 'Year_Built', 'Year_Remod_Add', 
                 'Kitchen_Qual', 'Gr_Liv_Area', 'Overall_Qual', 'Year_Sold']

# ==============================================================================
# 2. Handling Missing Values & Integrity Check
# ==============================================================================
print(f"Original dimensions: {ames_data.shape}")

# Drop rows with missing values in the selected columns
clean_ames_data = ames_data[selected_vars].dropna().copy()

print(f"Cleaned dimensions: {clean_ames_data.shape}")

# Export cleaned data
clean_ames_data.to_csv("clean_ames_data_for_analysis.csv", index=False)

# ==============================================================================
# 3. Deriving Variables
# ==============================================================================
# Create 'Renovation_Status'
clean_ames_data['Renovation_Status'] = np.where(
    clean_ames_data['Year_Remod_Add'] > clean_ames_data['Year_Built'], 
    "Remodeled", 
    "Not_Remodeled"
)

# Create 'House_Age' and 'House_Age_Adj'
clean_ames_data['House_Age'] = clean_ames_data['Year_Sold'] - clean_ames_data['Year_Built']
# Adjust House_Age to be at least 1 to avoid issues with log/box-cox if 0
clean_ames_data['House_Age_Adj'] = clean_ames_data['House_Age'].apply(lambda x: 1 if x <= 0 else x)

# Define Factor Levels (Ordering)
kitchen_qual_order = ["Po", "Fa", "TA", "Gd", "Ex"] # Raw codes: Poor, Fair, Typical/Avg, Good, Excellent
# Map raw codes to the labels used in R script if necessary, or just rely on order.
# The raw data uses abbreviations (Po, Fa, TA, Gd, Ex). The R script had full names.
# We will map them for consistency with the R output labels.
qual_map = {"Po": "Poor", "Fa": "Fair", "TA": "Typical", "Gd": "Good", "Ex": "Excellent"}
clean_ames_data['Kitchen_Qual'] = clean_ames_data['Kitchen_Qual'].map(qual_map)

# Define ordered categories
clean_ames_data['Kitchen_Qual'] = pd.Categorical(
    clean_ames_data['Kitchen_Qual'], 
    categories=["Poor", "Fair", "Typical", "Good", "Excellent"], 
    ordered=True
)

# Note: Overall_Qual is numeric (1-10) in raw data, but treated as factor in R.
# We will bin/map it to match the R script's levels if we want exact parity, 
# but simply converting to categorical is usually sufficient. 
# The R script levels: Very_Poor...Excellent.
# Mapping 1-10 to descriptions:
overall_qual_map = {
    1: "Very_Poor", 2: "Poor", 3: "Fair", 4: "Below_Average", 5: "Average",
    6: "Above_Average", 7: "Good", 8: "Very_Good", 9: "Excellent", 10: "Very_Excellent"
}
clean_ames_data['Overall_Qual'] = clean_ames_data['Overall_Qual'].map(overall_qual_map)
clean_ames_data['Overall_Qual'] = pd.Categorical(
    clean_ames_data['Overall_Qual'],
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
