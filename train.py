import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from models.alexnet import AlexNet
from utils.train_utils import train, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)


model = AlexNet().to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 10

for epoch in range(epochs):

    loss = train(model, train_loader, optimizer, criterion, device)

    acc = evaluate(model, test_loader, device)

    print(f"Epoch {epoch+1}/{epochs}")
    print("Loss:", loss)
    print("Accuracy:", acc)


torch.save(model.state_dict(), "alexnet_cifar10.pth")