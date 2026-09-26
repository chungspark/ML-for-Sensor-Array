"""Random-forest classification and regression of sensor-array responses.
CSV rows: gas type, concentration (ppm), S1, S2, S3, S4.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

DATA_FILE = "00_Sensor Array Responses.csv"
OUTPUT_FILE = "hierarchical_cv_prediction_comparison.csv"
RANDOM_STATE = 42
N_SPLITS = 5
N_TREES = 100

# 1. Load the four sensor responses and the gas/concentration labels.
data = pd.read_csv(DATA_FILE, header=None)
data = data.dropna(how="all", axis=0).dropna(how="all", axis=1).T
data.columns = ["GasType", "Concentration", "S1", "S2", "S3", "S4"]
for column in ["Concentration", "S1", "S2", "S3", "S4"]:
    data[column] = pd.to_numeric(data[column])

X = data[["S1", "S2", "S3", "S4"]].values
y_class = data["GasType"].values
y_reg = data["Concentration"].values

# 2. Define gas-stratified fivefold CV and the random-forest models.
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
classifier = RandomForestClassifier(
    n_estimators=N_TREES, random_state=RANDOM_STATE, class_weight="balanced"
)
regressor = RandomForestRegressor(
    n_estimators=N_TREES, random_state=RANDOM_STATE
)

# 3. Train only on each training fold and collect out-of-fold predictions.
y_class_pred = np.empty_like(y_class, dtype=object)
y_reg_pred = np.zeros_like(y_reg, dtype=float)

for train_idx, test_idx in cv.split(X, y_class):
    clf = clone(classifier)
    clf.fit(X[train_idx], y_class[train_idx])
    gas_predictions = clf.predict(X[test_idx])
    y_class_pred[test_idx] = gas_predictions

    gas_regressors = {}
    for gas in np.unique(y_class[train_idx]):
        gas_train_idx = train_idx[y_class[train_idx] == gas]
        reg = clone(regressor)
        reg.fit(X[gas_train_idx], y_reg[gas_train_idx])
        gas_regressors[gas] = reg

    # Route each sample using its predicted, not actual, gas identity.
    for sample_idx, predicted_gas in zip(test_idx, gas_predictions):
        y_reg_pred[sample_idx] = gas_regressors[predicted_gas].predict(
            X[sample_idx].reshape(1, -1)
        )[0]

# 4. Evaluate the pooled out-of-fold predictions.
accuracy = accuracy_score(y_class, y_class_pred)
mse = mean_squared_error(y_reg, y_reg_pred)
r2 = r2_score(y_reg, y_reg_pred)

print(classification_report(y_class, y_class_pred, digits=4))
print(f"Classification accuracy: {accuracy:.4%}")
print(f"Concentration MSE (ppm^2): {mse:.4f}")
print(f"Concentration R2: {r2:.4f}")

# 5. Save the predictions used to calculate the reported metrics.
predictions = pd.DataFrame({
    "GasType_actual": y_class,
    "GasType_predicted": y_class_pred,
    "Actual_Concentration": y_reg,
    "Predicted_Concentration": y_reg_pred,
    "Absolute_Error": np.abs(y_reg - y_reg_pred),
})
predictions.to_csv(OUTPUT_FILE, index=False)
print(f"Predictions saved to: {OUTPUT_FILE}")

