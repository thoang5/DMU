import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Generate data
x_numpy = np.linspace(0, 1, 100, dtype=np.float32)
y_numpy = (1 - x_numpy) * (np.sin(20 * np.log(x_numpy + 0.2))).astype(np.float32)

# Convert to tensor
x_tensor = torch.from_numpy(x_numpy).view(-1, 1)   # shape (100, 1)
y_tensor  = torch.from_numpy(y_numpy).view(-1, 1)   # shape (100, 1)

# Create the model
#n_samples, n_features = x.shape
#input_size = n_features
output_size = 1

degree = 6
X_poly = torch.cat([x_tensor**i for i in range(1, degree + 1)], dim=1)
#model = nn.Linear(6, output_size)
# model = nn.Sequential(
#     nn.Linear(6, 12),
#     nn.ReLU(),
#     nn.Linear(12, 12),
#     nn.ReLU(),
#     nn.Linear(12, 1)
# )

# Neural network: 1 -> 64 -> 64 -> 1
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

# Create loss and optimizer
learning_rate = 0.05
loss_function = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Train the model
epochs = 5000
losses = []
epoch_nums = []
for epoch in range(epochs):
    # forward pass
    y_predicted = model(x_tensor)
    loss = loss_function(y_predicted, y_tensor)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # saving epochs and losses
    losses.append(loss.item())
    epoch_nums.append(epoch+1)

    if (epoch + 1) % 10 == 0:
        print(f"epoch: {epoch + 1}, loss: {loss.item():.5f}")

# plot the curves
predicted = model(x_tensor).detach().numpy()

plt.figure(figsize=(10, 6))
plt.scatter(x_numpy, y_numpy, s=22, alpha=0.75, label="True data", color = 'r')
plt.plot(x_numpy, predicted, linewidth=2.5, label="Model prediction", color='b')

plt.title("Function Approximation with PyTorch", fontsize=16)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(8, 5))
plt.plot(epoch_nums, losses, linewidth=2)
plt.title("Epoch vs Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=True)