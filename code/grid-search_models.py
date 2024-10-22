from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import numpy as np

# Load dataset
df = pd.read_csv("data/football_matches.csv")

# Preprocess the data
features = df.drop(columns=['ID', 'season', 'date', 'goal_home_ft', 'goal_away_ft', 'sg_match_ft', 'result'])
target = df['result'].astype('category').cat.codes  # Convert categorical 'result' to numerical

# One-hot encode categorical features
features = pd.get_dummies(features, columns=['home_team', 'away_team']).astype(float)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(64,), (64, 32), (64, 32, 16),
                           (128,), (128, 64), (128, 64, 32),
                           (256,), (256, 128), (256, 128, 64)],  # Different hidden layer configurations
    'activation': ['relu', 'tanh'],  # Activation functions to try
    'solver': ['adam'],  # Optimizer (keep Adam as it's commonly used for neural networks)
    'learning_rate_init': [0.01, 0.001, 0.0001, 0.00001],  # Learning rates to try
    'alpha': [0.00001, 0.0001, 0.001],  # Regularization strength (L2 penalty)
    'batch_size': [8, 16, 32, 64, 128],  # Batch sizes
    'max_iter': [5, 10, 15, 20, 25],  # Number of iterations (epochs)
    'early_stopping': [True]  # Enable early stopping to avoid overfitting
}

# Initialize the MLPClassifier
mlp = MLPClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, 
                           cv=3,  # 3-fold cross-validation
                           verbose=2,  # Display training progress
                           n_jobs=-1,  # Use all available CPU cores
                           scoring='accuracy')  # Metric to optimize

# Fit the grid search model
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with Best Model: {test_accuracy:.4f}")
# Print classification report
print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

# Calculate permutation importance
perm_importance = permutation_importance(best_model, X_test_scaled, y_test, n_repeats=10, random_state=42)

# Display feature importance
feature_importances = perm_importance.importances_mean
sorted_idx = np.argsort(feature_importances)[::-1]  # Sort in descending order

print("Permutation Feature Importances:")
for idx in sorted_idx:
    print(f"Feature {X_train.columns[idx]}: Importance {feature_importances[idx]:.4f}")

conf_mtx = confusion_matrix(np.array(y_test), np.array(y_pred))
display = ConfusionMatrixDisplay(conf_mtx)
display.plot()
plt.show()