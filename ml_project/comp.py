import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load CSV data
data = pd.read_csv("california_housing_sample.csv")  # Make sure this file is in the same directory

# Split into features and target
X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]

# Check for constant values in the target column
if y.nunique() <= 1:
    raise ValueError("Target variable has only one unique value. R² cannot be calculated.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
}

# Store results for comparison
results = {'MSE': {}, 'R²': {}}

# Train and evaluate each model
for model_name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    try:
        r2 = r2_score(y_test, y_pred)
    except:
        r2 = float('nan')  # If R² cannot be calculated, set it to NaN
    
    # Save results
    results['MSE'][model_name] = mse
    results['R²'][model_name] = r2

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Plot comparison of models' performance
ax = results_df.plot(kind='bar', figsize=(10, 6), width=0.8)
plt.title('Model Performance Comparison (MSE and R² Score)')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
