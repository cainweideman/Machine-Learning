import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, log_loss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from collections import Counter


# Load dataset
df = pd.read_csv("football_matches.csv")

# Preprocess the data
features = df.drop(columns=['ID', 'season', 'date', 'goal_home_ft', 'goal_away_ft', 'sg_match_ft', 'result'])
#features = df.drop(columns=['ID', 'home_team', 'away_team', 'season', 'date', 'goal_home_ft', 'goal_away_ft', 'sg_match_ft', 'result'])
target = df['result'].astype('category').cat.codes  # Convert categorical 'result' to numerical

# One-hot encode categorical features
features = pd.get_dummies(features, columns=['home_team', 'away_team']).astype(float)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the MLPClassifier model
model = MLPClassifier(hidden_layer_sizes=(256, 128), 
                      activation='tanh',
                      max_iter=10,  # Increase iterations
                      batch_size=128,  # Same batch size as PyTorch model
                      solver='adam', 
                      learning_rate_init=0.001,  # Same learning rate as PyTorch
                      alpha=0.00001,  # Reduce regularization strength
                      random_state=42,
                      early_stopping=True  # Optional: can prevent overfitting
                     )

# Train the model
model.fit(X_train_scaled, y_train)

# Print the loss after training
print(f"Training loss: {model.loss_:.4f}")
print("Training accuracy: ", model.score(X_train_scaled, y_train))

# Evaluate the model
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)
test_loss = log_loss(y_test, y_pred_proba)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test set log loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Print classification report
print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

# Calculate permutation importance
perm_importance = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)

# Display feature importance
feature_importances = perm_importance.importances_mean
sorted_idx = np.argsort(feature_importances)[::-1]  # Sort in descending order

print("Permutation Feature Importances:")
for idx in sorted_idx:
    print(f"Feature {X_train.columns[idx]}: Importance {feature_importances[idx]:.4f}")

conf_mtx = confusion_matrix(np.array(y_test), np.array(y_pred))
display = ConfusionMatrixDisplay(conf_mtx, display_labels=['Home Win', 'Away Win', 'Draw'])
display.plot()
plt.show()

print(Counter(y_test))