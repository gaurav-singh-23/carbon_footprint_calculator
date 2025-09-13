# notebooks/train_model.py
import sys, os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Make root folder accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from functions import input_preprocessing  # preprocessing function

# =======================
# Load Dataset
# =======================
df = pd.read_csv("data/Carbon_Emission_Smart.csv")

# Separate features/target
X = df.drop(columns=["CarbonEmission"])
y = df["CarbonEmission"]

# Apply preprocessing (encoding categorical vars)
X = input_preprocessing(X)

# Align with sample template (to keep consistent columns)
from functions import sample
sample_df = pd.DataFrame(data=sample, index=[0])
sample_df[sample_df.columns] = 0
for col in X.columns:
    if col not in sample_df.columns:
        sample_df[col] = 0
X = X.reindex(columns=sample_df.columns, fill_value=0)

# =======================
# Train/Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================
# Define Models
# =======================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "MLP Regressor": MLPRegressor(hidden_layer_sizes=(64, 32), 
                                  max_iter=500, 
                                  random_state=42)
}

best_model = None
best_score = -999

# =======================
# Train & Evaluate
# =======================
for name, model in models.items():
    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n{name}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")

    if r2 > best_score:
        best_score = r2
        best_model = model

# =======================
# Save Best Model
# =======================
os.makedirs("models", exist_ok=True)
pickle.dump(best_model, open("models/model.sav", "wb"))
pickle.dump(scaler, open("models/scale.sav", "wb"))

print(f"\n✅ Best model saved: {type(best_model).__name__} (R² = {best_score:.4f})")
