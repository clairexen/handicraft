import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Configuration
dataFile = "ptdata.txt"
y_min, y_max = 1, 8
hiddenLayers = (500,)
batch_size = 1024
epochs = 8
cycles = 8
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
            #else:
            #    layers.append(nn.Softplus())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

print("Loading...")

# Function to load dataset from file
def load_dataset(file_path):
    bindata = np.fromfile(file_path, dtype=np.byte)
    recordSize = np.where(bindata[:100000] == ord('\n'))[0][0]
    bindata = bindata.reshape(-1, recordSize+1)
    data = np.array(bindata[:,:recordSize-1] - ord('0'), dtype=np.float32)
    labels = np.array(bindata[:,recordSize-1] - ord('A'), dtype=np.float32).reshape(-1, 1)
    return torch.tensor(data), torch.tensor(labels)

X, y = load_dataset(dataFile)
if y_min is None: y_min = torch.min(y).item()
if y_max is None: y_max = torch.max(y).item()
y = (y - y_min) / (y_max - y_min)
dataset = TensorDataset(X, y)
num_samples = X.shape[0]
print(f"Dataset size: {num_samples} samples")

# ANN Dimensions
input_size = X.shape[1]
sz = (input_size,) + hiddenLayers + (1,)
print(f"ANN model geometry: {sz}")

# Split dataset into training and test sets
train_size = int(train_ratio * num_samples)
test_size = num_samples - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device_type = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Setting up on device \"{device_type}\"...")

# Initialize model, loss function, and optimizer
device = torch.device(device_type)
model = ConfigurableANN(sz).to(device)
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for cycle in range(cycles):
    print()
    print(f"== CYCLE {cycle+1}/{cycles} ==")

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
    correct02 = 0
    correct05 = 0
    correct08 = 0
    deltasum = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            test_loss += loss.item()

            # Accuracy calculation
            correct02 += ((predictions - batch_y).abs() < 0.2/(y_max-y_min)).sum().item()
            correct05 += ((predictions - batch_y).abs() < 0.5/(y_max-y_min)).sum().item()
            correct08 += ((predictions - batch_y).abs() < 0.8/(y_max-y_min)).sum().item()
            deltasum += (predictions - batch_y).abs().sum().item()
            total += batch_y.size(0)

    print(f"Test Loss: {test_loss/len(test_loader):.6f}")
    print(f"Test Mean Abs Delta: {(y_max-y_min) * deltasum / total:.2f}")
    print(f"Test Accuracy: {100 * correct02 / total:.2f}% " +
          f"{100 * correct05 / total:.2f}% {100 * correct08 / total:.2f}%")

print()
print("Evaluating...")

# Function to make a prediction on a single game state
def predict_moves(game_state):
    game_state = torch.tensor(game_state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(game_state)
        return output.item() * (y_max - y_min) + y_min

deltas = dict()
with open(dataFile, "r") as file:
    for line in file:
        line = line.strip()
        fIn = [int(c) for c in line[:-1]]
        fOut = ord(line[-1])-ord('A')
        out = predict_moves(fIn)
        if fOut not in deltas:
            deltas[fOut] = list()
        if len(deltas[fOut]) == 200:
            break
        deltas[fOut].append(out - fOut)

for depth in sorted(deltas.keys()):
    if len(deltas[depth]) < 10:
        continue
    v = np.array(deltas[depth])
    Min, Max, Avg, Std = np.min(v), np.max(v), np.mean(v), np.std(v)
    print(f"Results for depth {depth:2d} (N = {len(v):3d}): " +
          f"{Min=:+.2f} {Max=:+.2f} {Avg=:+.2f} ({Std=:+.2f})")

