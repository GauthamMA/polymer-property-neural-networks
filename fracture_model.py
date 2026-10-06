# Fracture toughness neural-network demonstration
#
# (worksheet: fracture_model). Requires pandas, openpyxl, matplotlib,
# and scikit-learn. Prints results and saves the prediction plot in that folder.
#
# Results describe a single split with material grades shared across both sets;
# they do not establish performance on completely unfamiliar material grades.

# Step 1
# Import pandas for tabular data and matplotlib for plotting.
#importing libraries

import pandas as pd
import matplotlib.pyplot as plt

# Step 2
# Load the Excel worksheet from the current working directory.
# Show the dataset size, first five rows, column types, and non-missing counts.
#loading data

df = pd.read_excel('fracture_toughness_model.xlsx', sheet_name = 'fracture_model')
print(df.shape)
print(df.head())
df.info()

# Step 3
# Separate the target (y), input features (X), and material–grade group labels.
# Group labels are used to balance the split; the listed identifiers are excluded
# from the model inputs. Polymer material remains an input feature.
target = "Mode I fracture toughness, K_Ic — recorded (MPa√m)"

y = df[target].copy()

groups = df["Polymer material and grade group"].copy()

X = df.drop(columns=[
    target,
    "Source record ID",
    "Polymer material and grade group",
    "Material grade",
    "Manufacturer product / brand name",
    "Flexural elastic modulus — unit and evidence status"
]).copy()

print("Input shape:", X.shape)
print("Target shape:", y.shape)
print("Number of material groups:", groups.nunique())

# Step 4
# Reserve 20% of rows for testing and use 80% for training.
# Stratifying by group keeps roughly the same material–grade proportions in both
# sets. This evaluates new observations from represented grades, not unseen grades.
# The fixed random seed makes the split reproducible. Splitting all three objects
# together keeps inputs, targets, and group labels aligned. Copy inputs before editing.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X,
    y,
    groups,
    test_size=0.20,
    random_state=42,
    stratify=groups
)

X_train = X_train.copy()
X_test = X_test.copy()

print("Training inputs:", X_train.shape)
print("Test inputs:", X_test.shape)

print("\nTraining groups:", groups_train.nunique())
print("Test groups:", groups_test.nunique())

print("\nTest groups and row counts:")
print(groups_test.value_counts())

# Step 5
# Replace missing numeric values with the median of each training column.
# Apply those SAME training medians to the test set to avoid learning from test data.
numeric_columns = X_train.select_dtypes(
    include="number"
).columns

training_medians = X_train[numeric_columns].median()

X_train[numeric_columns] = (
    X_train[numeric_columns].fillna(training_medians)
)

X_test[numeric_columns] = (
    X_test[numeric_columns].fillna(training_medians)
)

# Step 6
# Check that numeric imputation succeeded. Both counts should be zero.
# An entirely missing training column would have no usable median.
print(
    "Missing numeric training cells:",
    X_train[numeric_columns].isna().sum().sum()
)

print(
    "Missing numeric test cells:",
    X_test[numeric_columns].isna().sum().sum()
)

# Step 7
# Find non-numeric input columns and one-hot encode their categories.
# Each category becomes a numeric indicator column (0 or 1).
categorical_columns = X_train.select_dtypes(
    exclude="number"
).columns

print(categorical_columns)

X_train_encoded = pd.get_dummies(
    X_train,
    columns=categorical_columns,
    dtype=float
)

X_test_encoded = pd.get_dummies(
    X_test,
    columns=categorical_columns,
    dtype=float
)

# Step 8
# Match the test columns and their order to the training columns.
# Add missing indicator columns with zeros and discard test-only columns.
X_test_encoded = X_test_encoded.reindex(
    columns=X_train_encoded.columns,
    fill_value=0
)

print("Encoded training shape:", X_train_encoded.shape)
print("Encoded test shape:", X_test_encoded.shape)

# Step 9
# Standardize numeric inputs using training-set means and standard deviations.
# Use the fitted scaler for both sets; do not fit it again on test data.
# One-hot indicator columns are left unchanged.
from sklearn.preprocessing import StandardScaler

X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()

scaler = StandardScaler()

scaler.fit(X_train_encoded[numeric_columns])

X_train_scaled[numeric_columns] = scaler.transform(
    X_train_encoded[numeric_columns]
)

X_test_scaled[numeric_columns] = scaler.transform(
    X_test_encoded[numeric_columns]
)

# Step 10
# Create and fit a neural network with one hidden layer containing 10 neurons.
# tanh is the hidden-layer activation; lbfgs optimizes the model weights.
# max_iter caps optimizer iterations; it does not guarantee convergence.
# random_state fixes the random initialization. The regression output is
# unconstrained, so negative predictions are possible even for physical toughness.
from sklearn.neural_network import MLPRegressor

neural_model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation="tanh",
    solver="lbfgs",
    max_iter=10000,
    random_state=42
)

neural_model.fit(X_train_scaled, y_train)

# Step 11
# Predict toughness for training and held-out test observations.
# Targets were not scaled, so predictions are already in MPa√m.
train_predictions = neural_model.predict(X_train_scaled)
test_predictions = neural_model.predict(X_test_scaled)

# Step 12
# Compare the first ten test predictions with their measured values.
# Keep original row indices so each observation can be traced to the dataset.
# print replaces the notebook-only display call.
prediction_comparison = pd.DataFrame({
    "Measured K_Ic (MPa√m)": y_test.to_numpy(),
    "Predicted K_Ic (MPa√m)": test_predictions
}, index=y_test.index)

print(prediction_comparison.head(10))

# Step 13
# Mean absolute error (MAE) is the average magnitude of prediction errors.
# It has the target units (MPa√m); lower is better.
from sklearn.metrics import mean_absolute_error

train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)

print("Training MAE (MPa√m):", train_mae)
print("Test MAE (MPa√m):", test_mae)

# Step 14
# R² compares squared prediction errors with deviations from the evaluated
# set’s mean. A score of 1 is perfect; 0 matches predicting that mean;
# negative scores are worse. Compare training and test scores for a performance gap.
from sklearn.metrics import r2_score

train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

print("Training R²:", train_r2)
print("Test R²:", test_r2)

# Step 15
# Plot measured versus predicted test values. The dashed line represents
# perfect predictions; points above it are overpredictions and below it are
# underpredictions. Save a PNG in the current working directory.
# In a standalone script, close the graph window to continue to the next section.
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))

plt.scatter(y_test, test_predictions, alpha=0.6)

lower = min(y_test.min(), test_predictions.min())
upper = max(y_test.max(), test_predictions.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    "r--",
    label="Perfect prediction"
)

plt.xlabel("Measured fracture toughness (MPa√m)")
plt.ylabel("Predicted fracture toughness (MPa√m)")
plt.title("Fracture toughness: test predictions")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("fracture_toughness_predictions.png", dpi=300)
plt.show()

# Step 16
# Inspect the ten largest absolute test errors and their material–grade groups.
# This is a diagnostic table only: it does not remove rows or change the model.
comparison = X_test.copy()

comparison["Measured K_Ic"] = y_test
comparison["Predicted K_Ic"] = test_predictions
comparison["Absolute error"] = abs(y_test - test_predictions)
comparison["Material-grade group"] = groups_test

print(
    comparison[
        [
            "Material-grade group",
            "Measured K_Ic",
            "Predicted K_Ic",
            "Absolute error"
        ]
    ]
    .sort_values("Absolute error", ascending=False)
    .head(10)
    .to_string()
)
