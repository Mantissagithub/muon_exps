# script to train a simple dense mlp on mnist dataset with the muon optimizer
import torch
import torch.nn as nn
from torch.optim import Muon, AdamW

# Epoch 30/30, Loss: 0.0170
# Test Accuracy: 97.48%

class SimpleMLP(nn.Module):
  def __init__(self, lr=1e-3, weight_decay=0.1):
      super(SimpleMLP, self).__init__()
      self.fc1 = nn.Linear(28*28, 256)
      self.fc2 = nn.Linear(256, 128)
      self.fc3 = nn.Linear(128, 10)
      self.optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
      self.criterion = nn.CrossEntropyLoss()

  def forward(self, x):
      x = x.view(-1, 28*28)
      x = torch.relu(self.fc1(x))
      x = torch.relu(self.fc2(x))
      x = self.fc3(x)
      return x

def train_step(model, data, target):
    model.optimizer.zero_grad()
    output = model(data)
    loss = model.criterion(output, target)
    loss.backward()
    model.optimizer.step()
    return loss.item()

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP().to(device)
    num_epochs = 30

    losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            loss = train_step(model, data, target)
            losses.append(loss)
            total_loss += loss
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    model.eval()

    # visualizing loss curve
    import matplotlib.pyplot as plt
    import os

    os.makedirs('results/loss_curves', exist_ok=True)
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve with AdamW Optimizer')
    plt.savefig('results/loss_curves/mnist_muon_loss_curve_with_adamw.png')
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.2f}%")

