import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.alexnet import AlexNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=128)


model = AlexNet().to(device)

model.load_state_dict(torch.load("alexnet_cifar10.pth"))

model.eval()


correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print("Test Accuracy:", 100 * correct / total)