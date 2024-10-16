import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn


# Set seed
torch.manual_seed(42)

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
df = pd.read_csv("football_matches.csv")
#print(df.head())

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

features = df.drop(columns=['ID', 'season', 'date', 'goal_home_ft', 'goal_away_ft', 'sg_match_ft', 'result'])
features['home_team'] = label_encoder.fit_transform(features['home_team'])
features['away_team'] = label_encoder.fit_transform(features['away_team'])

target = df['result']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the data to PyTorch tensors and move them to the GPU (if available)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)  # Use long for classification
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

# Define a simple neural network model for classification in PyTorch
class MatchOutcomePredictor(nn.Module):
    def __init__(self, input_dim):
        super(MatchOutcomePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Output layer (3 outputs: home win, away win, draw)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model and move it to the GPU (if available)
input_dim = X_train_tensor.shape[1]
model = MatchOutcomePredictor(input_dim).to(device)

# Print the model to see its architecture
print(model)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Training parameters
num_epochs = 20  # Number of training epochs
batch_size = 16  # Size of the batches

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i in range(0, X_train_tensor.size(0), batch_size):
        # Get the batch of data
        X_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)

        # Compute the loss
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
        total_train += y_batch.size(0)
        correct_train += (predicted == y_batch).sum().item()

    # Calculate average loss and accuracy for the training set
    avg_loss = running_loss / (X_train_tensor.size(0) // batch_size)
    train_accuracy = correct_train / total_train * 100

    # Print the training loss and accuracy for the epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')

# Evaluation on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    # Get the predictions for the test set
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs.data, 1)  # Get the index of the max log-probability

    # Calculate testing accuracy
    correct_test = (predicted == y_test_tensor).sum().item()
    total_test = y_test_tensor.size(0)
    test_accuracy = correct_test / total_test * 100

    print(f'Accuracy of the model on the test set: {test_accuracy:.2f}%')