import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Set seed
torch.manual_seed(42)

# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Load the dataset
df = pd.read_csv("football_matches.csv")
features = df.drop(columns=['ID', 'season', 'date', 'goal_home_ft', 'goal_away_ft', 'sg_match_ft', 'result'])
target = df['result']
print(Counter(target))

# One-hot encode categorical features
one_hot_encoded_features = pd.get_dummies(features, columns=['home_team', 'away_team']).astype(int)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(one_hot_encoded_features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

# Define the neural network model
class MatchOutcomePredictor(nn.Module):
    def __init__(self, input_dim):
        super(MatchOutcomePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
input_dim = X_train_tensor.shape[1]
model = MatchOutcomePredictor(input_dim).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training parameters
num_epochs = 20
batch_size = 32

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i in range(0, X_train_tensor.size(0), batch_size):
        X_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += y_batch.size(0)
        correct_train += (predicted == y_batch).sum().item()

    avg_loss = running_loss / (X_train_tensor.size(0) // batch_size)
    train_accuracy = correct_train / total_train * 100
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')

# Step 1: Create a wrapper for the PyTorch model
class PyTorchClassifier:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def fit(self, X, y):
        # Fit method can be left empty if already trained
        pass

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()

    def score(self, X, y):
        """Compute the accuracy of the model on the provided data."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

# Step 2: Use the wrapper to compute permutation importance
pytorch_model = PyTorchClassifier(model, device)

# Calculate baseline accuracy
baseline_accuracy = accuracy_score(y_test, pytorch_model.predict(X_test_scaled))
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

# Compute permutation importance
result = permutation_importance(
    estimator=pytorch_model,
    X=X_test_scaled,
    y=y_test,
    n_repeats=30,
    random_state=42,
)

# Display feature importance
for i in result.importances_mean.argsort()[::-1]:
    print(f"{X_train.columns[i]}: {result.importances_mean[i]:.3f} ± {result.importances_std[i]:.3f}")