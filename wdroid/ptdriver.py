import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import struct

# Configuration
srcPthFile = None # "ptmodel.pth"
srcDatFiles = [f"ptdata{i}.txt" for i in range(8)]
evalDatFile = "ptdata0.txt"
outPrefix = "ptmodel"
y_min, y_max = 1, 8
hiddenLayers = (200,)
batch_size = 1024
num_epochs = 4
learning_rate = 0.001
train_ratio = 0.8  # 80% training, 20% testing


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

def load_dataset(file_path):
    print(f"Loading '{file_path}'...")
    bindata = np.fromfile(file_path, dtype=np.byte)
    recordSize = np.where(bindata[:100000] == ord('\n'))[0][0]
    bindata = bindata.reshape(-1, recordSize+1)
    data = np.array(bindata[:,:recordSize-1] - ord('0'), dtype=np.float32)
    labels = np.array(bindata[:,recordSize-1] - ord('A'), dtype=np.float32).reshape(-1, 1)
    print(f"Samples in dataset: {len(labels)}")
    return torch.tensor(data), torch.tensor(labels)

if srcPthFile is not None:
    print(f"Restoring '{srcPthFile}'...")
    srcState = torch.load(srcPthFile)
    hiddenLayers = (srcState["model.0.weight"].shape[0],)
    input_size = srcState["model.0.weight"].shape[1]
    sz = (input_size,) + hiddenLayers + (1,)
    X, y = None, None

else:
    srcState = None
    X, y = load_dataset(srcDatFiles[0])
    y = (y - y_min) / (y_max - y_min)
    dataset = TensorDataset(X, y)
    num_samples, input_size = X.shape
    sz = (input_size,) + hiddenLayers + (1,)

    train_size = int(train_ratio * num_samples)
    test_size = num_samples - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"ANN model geometry: {sz}")
device_type = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Setting up on device \"{device_type}\"...")

# Initialize model, loss function, and optimizer
device = torch.device(device_type)
model = ConfigurableANN(sz).to(device)
if srcState is not None:
    model.load_state_dict(srcState)
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for cycle, srcFile in enumerate(srcDatFiles):
    print()
    print(f"== CYCLE {cycle+1} ==")

    if cycle > 0 or srcState is not None:
        X, y = load_dataset(srcFile)
        y = (y - y_min) / (y_max - y_min)
        dataset = TensorDataset(X, y)
        num_samples = X.shape[0]
        train_size = int(train_ratio * num_samples)
        test_size = num_samples - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Training...")

    for epoch in range(num_epochs):
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

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.6f}")

    print("Testing...")

    model.eval()
    test_loss = 0
    correct02 = 0
    correct05 = 0
    correct08 = 0
    correct12 = 0
    correct15 = 0
    deltasum = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            test_loss += loss.item()

            correct02 += ((predictions - batch_y).abs() < 0.2/(y_max-y_min)).sum().item()
            correct05 += ((predictions - batch_y).abs() < 0.5/(y_max-y_min)).sum().item()
            correct08 += ((predictions - batch_y).abs() < 0.8/(y_max-y_min)).sum().item()
            correct12 += ((predictions - batch_y).abs() < 1.2/(y_max-y_min)).sum().item()
            correct15 += ((predictions - batch_y).abs() < 1.5/(y_max-y_min)).sum().item()

            deltasum += (predictions - batch_y).abs().sum().item()
            total += batch_y.size(0)

    print(f"Test Loss: {test_loss/len(test_loader):.6f}")
    print(f"Test Mean Abs Delta: {(y_max-y_min) * deltasum / total:.2f}")
    print(f"Test Accuracy: {100 * correct02 / total:.2f}% {100 * correct05 / total:.2f}% " +
          f"({100 * correct08 / total:.2f}% {100 * correct12 / total:.2f}% {100 * correct15 / total:.2f}%)")

print()
print("Exporting...")

def run_model(indata):
    indata = torch.tensor(indata, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(indata).item() * (y_max - y_min) + y_min

if X is not None:
    example_input = [v.item() for v in X[4,:]]
else:
    example_input = [0.0] * input_size
    example_input[0] = 1.0

example_inidx = [i+1 for i, v in enumerate(example_input) if v > 0.5]

example_layers = list()
hookHandles = list()

def fwdHook(module, args, output):
    example_layers.append((module, output))

for module in model.model.children():
    hookHandles.append(module.register_forward_hook(fwdHook))

example_output = run_model(example_input)

for handle in hookHandles:
    handle.remove()

if True:
    torch.save(model.state_dict(), f"{outPrefix}.pth")

if False:
    with open(f"{outPrefix}.hh", "w") as fh:
        with open(f"{outPrefix}.cc", "w") as fcc:
            fh.write(f"#ifndef {outPrefix.upper()}_HH\n")
            fh.write(f"#define {outPrefix.upper()}_HH\n")
            fh.write(f"#define WordleDroidANN_Dim0 {sz[0]}\n")
            fh.write(f"#define WordleDroidANN_Dim1 {sz[1]}\n")
            fcc.write(f"#include \"{outPrefix}.hh\"\n")
            for name, param in model.named_parameters():
                w = param.detach().numpy().transpose().flatten()
                fh.write(f"extern const float WordleDroidANN_{name.replace('.', '_')}[{len(w)}]; // {name}\n")
                fcc.write(f"const float WordleDroidANN_{name.replace('.', '_')}[{len(w)}] = {{")
                fcc.write(", ".join(map(str, w)))
                fcc.write("};\n")
            fh.write(f"#endif\n")

if True:
    with open(f"{outPrefix}.bin", "wb") as f:
        # ANN Dimensions
        f.write(struct.pack('i', sz[0]))
        f.write(struct.pack('i', sz[1]))
        f.flush()

        # Parameters
        keys = "model.0.weight model.0.bias model.2.weight model.2.bias".split()
        sizes = (sz[0]*sz[1], sz[1], sz[1], 1)
        for key, s in zip(keys, sizes):
            data = model.get_parameter(key).detach().numpy()
            data = data.transpose().flatten().astype(np.float32)
            assert len(data) == s
            data.tofile(f)

        # Test Vector
        f.write(struct.pack('i', len(example_inidx)))
        for idx in example_inidx:
            f.write(struct.pack('i', idx))
        f.flush()
        for _, data in example_layers:
            data.numpy().flatten().astype(np.float32).tofile(f)
        f.write(struct.pack('f', example_output))

if evalDatFile is not None:
    print("Evaluating...")

    deltas = dict()
    with open(evalDatFile, "r") as file:
        for line in file:
            line = line.strip()
            fIn = [int(c) for c in line[:-1]]
            fOut = ord(line[-1])-ord('A')
            out = run_model(fIn)
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
