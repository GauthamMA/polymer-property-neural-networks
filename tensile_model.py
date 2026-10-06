# Tensile strength neural-network demonstration
#
# (worksheet: Tensile_model). Requires pandas, openpyxl, matplotlib,
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

df = pd.read_excel('tensile_model.xlsx', sheet_name = 'Tensile_model')
print(df.shape)
print(df.head())
df.info()

# Step 3
# Separate the target (y), input features (X), and material–grade group labels.
# Group labels are used to balance the split; the listed identifiers are excluded
# from the model inputs. Polymer material remains an input feature.
target = "Applied stress, σ (MPa)"

y = df[target].copy()

groups = df["Polymer material and grade group"].copy()

X = df.drop(columns=[
    target,
    "Source record ID",
    "Polymer material and grade group",
    "Material grade",
    "Manufacturer product / brand name"
]).copy()

print("Input shape:", X.shape)
print("Target shape:", y.shape)
print("Elastic modulus included:", "Elastic modulus, E" in X.columns)

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

# Step 9
# Print the encoded shapes and preview two polymer indicator columns.
# print replaces the notebook-only display call.
print("Encoded training shape:", X_train_encoded.shape)
print("Encoded test shape:", X_test_encoded.shape)

print(
    X_train_encoded[
        ["Polymer material_ABS", "Polymer material_PC"]
    ].head()
)

# Step 10
# Copy the encoded inputs and create the numeric feature scaler.
X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Step 11
# Fit the scaler using training numeric columns only, then transform both sets.
# Numeric features are centered and scaled; one-hot indicators remain unchanged.
scaler.fit(X_train_encoded[numeric_columns])

X_train_scaled[numeric_columns] = scaler.transform(X_train_encoded[numeric_columns])
X_test_scaled[numeric_columns] = scaler.transform(X_test_encoded[numeric_columns])

# Step 12
# Create a neural network with one hidden layer containing 10 neurons.
# tanh is the hidden-layer activation; lbfgs optimizes model weights.
# The initial iteration limit is 2,000, exactly as in the notebook.
# random_state fixes the random initialization.
from sklearn.neural_network import MLPRegressor

neural_model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation="tanh",
    solver="lbfgs",
    max_iter=2000,
    random_state=42
)

# Step 13
# Run the original first training call with the 2,000-iteration limit.

neural_model.fit(X_train_scaled, y_train)

# Step 14
# Preserve the notebook’s second training call with a 10,000-iteration limit.
# With the default warm_start=False, this fit starts training again rather than
# continuing the previous fit. Subsequent predictions use this second fitted model.
# The higher limit does not guarantee convergence; check any optimizer warnings.
neural_model.set_params(max_iter=10000)

neural_model.fit(X_train_scaled, y_train)

# Step 15
# Predict tensile stress for training and held-out test observations.
# Targets were not scaled, so predictions are already in MPa.
train_predictions = neural_model.predict(X_train_scaled)

test_predictions = neural_model.predict(X_test_scaled)

# Step 16
# Compare the first ten test predictions with measured stresses, retaining
# original row indices. print replaces the notebook-only display call.
prediction_comparison = pd.DataFrame({
    "Measured stress (MPa)": y_test.to_numpy(),
    "Predicted stress (MPa)": test_predictions
}, index=y_test.index)

print(prediction_comparison.head(10))

# Step 17
# Mean absolute error (MAE) is the average magnitude of prediction errors
# in MPa. Lower values indicate smaller average errors.
from sklearn.metrics import mean_absolute_error

train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)

print("Training MAE (MPa):", train_mae)
print("Test MAE (MPa):", test_mae)

# Step 18
# Plot measured versus predicted test stress. Points on the dashed line
# are perfect predictions. Save the PNG using the original notebook filename.
# Close the graph window to let the standalone script print the remaining results.
plt.scatter(y_test, test_predictions, alpha=0.7)

lower = min(y_test.min(), test_predictions.min())
upper = max(y_test.max(), test_predictions.max())

plt.plot([lower, upper], [lower, upper], "r--")

plt.xlabel("Measured stress (MPa)")
plt.ylabel("Predicted stress (MPa)")
plt.title("Tensile model: measured vs predicted")
plt.savefig("ftensile_strength_predictions.png", dpi=300)

plt.tight_layout()
plt.show()

# Step 19
# R² measures performance relative to predicting the evaluated set’s mean.
# A score of 1 is perfect, 0 matches that baseline, and negative values are worse.
# Compare training and test scores to assess the performance gap on this split.
from sklearn.metrics import r2_score

train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

print("Training R²:", train_r2)
print("Test R²:", test_r2)

# Step 20
# Inspect the ten largest absolute test errors and their material–grade groups.
# This diagnostic table does not remove observations or retrain the model.
comparison = X_test.copy()

comparison["Measured tensile strength (MPa)"] = y_test
comparison["Predicted tensile strength (MPa)"] = test_predictions
comparison["Absolute error (MPa)"] = abs(y_test - test_predictions)
comparison["Material-grade group"] = groups_test

print(
    comparison[
        [
            "Material-grade group",
            "Measured tensile strength (MPa)",
            "Predicted tensile strength (MPa)",
            "Absolute error (MPa)"
        ]
    ]
    .sort_values("Absolute error (MPa)", ascending=False)
    .head(10)
    .to_string()
)
