import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Configuration
doRegression = True
hiddenLayers = (500, 500)
batch_size = 64
epochs = 8
cycles = 2
learning_rate = 0.001
train_ratio = 0.8  # 80% training, 20% testing


# Define the neural network with a configurable number of hidden layers
class ConfigurableANN(nn.Module):
    def __init__(self, sz):
        super(ConfigurableANN, self).__init__()

        layers = []
        for i in range(len(sz)-1):
            layers.append(nn.Linear(sz[i], sz[i+1]))
            if i < len(sz)-2:
                layers.append(nn.ReLU())
            elif sz[i+1] == 1:
                layers.append(nn.Softplus())
            else:
                layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Function to load dataset from file
def load_dataset(file_path):
    data, labels = [], []
    with open(file_path, "r") as file:
        for line in file:
            [fIn, fOut] = line.strip().split()
            data.append(np.array([int(x) for x in fIn], dtype=np.float32))
            if doRegression:
                labels.append(np.array([int(fOut)], dtype=np.float32))
            else:
                labels.append(int(fOut)-1)
    data = torch.tensor(np.array(data))
    if doRegression:
        labels = torch.tensor(np.array(labels))
    else:
        labels = torch.tensor(np.array(labels), dtype=torch.long)
    return data, labels

print("Loading...")

X, y = load_dataset("ptdata.txt")
if doRegression:
    y_min = torch.min(y).item()
    y_max = torch.max(y).item()
    y = (y - y_min) / (y_max - y_min)
dataset = TensorDataset(X, y)
num_samples = X.shape[0]

# ANN Dimensions
input_size = X.shape[1]
output_size = 1 if doRegression else torch.max(y).item()+1
sz = (input_size,) + hiddenLayers + (output_size,)

# Split dataset into training and test sets
train_size = int(train_ratio * num_samples)
test_size = num_samples - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
device_type = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Requesting Device: {device_type}")
device = torch.device(device_type)
model = ConfigurableANN(sz).to(device)
if doRegression:
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
else:
    criterion = nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for cycle in range(cycles):
    print("")
    print(f"== CYCLE {cycle+1} ==")

    print("Training...")

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")

    print("Testing...")

    # Evaluation on test data
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            test_loss += loss.item()

            # Accuracy calculation
            predicted_classes = torch.argmax(predictions, dim=1)
            correct += (predicted_classes == batch_y).sum().item()
            total += batch_y.size(0)

    print(f"Test Loss: {test_loss/len(test_loader):.6f}")
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Function to make a prediction on a single game state
def predict_moves(game_state):
    game_state = torch.tensor(game_state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(game_state)
        if doRegression:
            return output.item() * (y_max - y_min) + y_min
        if True:
            return torch.argmax(output, dim=1).item() + 1
        return " ".join([f"{i+1}:{float(v):4.2f}" for i, v in enumerate(output[0])])

print("")
x = "100110111100010111011010000011010111000000000110001111001000111001111000001110001011001000"
print(f"Example Input State: {x}")
x = [int(c) for c in x]
print(f"Predicted remaining moves: {predict_moves(x)}")
