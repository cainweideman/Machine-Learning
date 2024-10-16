import pandas as pd
import itertools
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
from collections import Counter


# Set seed
torch.manual_seed(42)

# If there's a GPU available...
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

df = pd.read_csv("football_matches.csv")

features = df.drop(columns=['ID', 'season', 'date', 'goal_home_ft', 'goal_away_ft', 'sg_match_ft', 'result'])
target = df['result']

one_hot_encoded_features = pd.get_dummies(features, columns=['home_team', 'away_team']).astype(int)
print(one_hot_encoded_features)

X_train, X_test, y_train, y_test = train_test_split(one_hot_encoded_features, target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

class MatchOutcomePredictor(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes):
        super(MatchOutcomePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layer_sizes[0])
        self.fc2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.fc3 = nn.Linear(hidden_layer_sizes[1], 3)  # 3 classes: home win, away win, draw

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_and_evaluate_model(hidden_layer_sizes, learning_rate, num_epochs, batch_size):
    input_dim = X_train_tensor.shape[1]
    model = MatchOutcomePredictor(input_dim, hidden_layer_sizes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        correct_test = (predicted == y_test_tensor).sum().item()
        total_test = y_test_tensor.size(0)
        test_accuracy = correct_test / total_test * 100

    return test_accuracy

# Hyperparameters tuning ground
hidden_layer_sizes_options = [(64, 32), (128, 64), (256, 128)]
learning_rate_options = [0.01, 0.001, 0.0001, 0.00001]
num_epochs_options = [5, 10, 15, 20, 25]
batch_size_options = [8, 16, 32, 64, 128]

best_accuracy = 0.0
best_hyperparameters = None

for hidden_layer_sizes, learning_rate, num_epochs, batch_size in itertools.product(
        hidden_layer_sizes_options, learning_rate_options, num_epochs_options, batch_size_options):

    print(f"Training with hidden_layer_sizes={hidden_layer_sizes}, learning_rate={learning_rate}, "
          f"num_epochs={num_epochs}, batch_size={batch_size}")

    accuracy = train_and_evaluate_model(hidden_layer_sizes, learning_rate, num_epochs, batch_size)

    print(f"Test Accuracy: {accuracy:.4f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_hyperparameters = (hidden_layer_sizes, learning_rate, num_epochs, batch_size)

print(f"Best Test Accuracy: {best_accuracy:.4f}%")
print(f"Best Hyperparameters: hidden_layer_sizes={best_hyperparameters[0]}, learning_rate={best_hyperparameters[1]}, "
      f"num_epochs={best_hyperparameters[2]}, batch_size={best_hyperparameters[3]}")
