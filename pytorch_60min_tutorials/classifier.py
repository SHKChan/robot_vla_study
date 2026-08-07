from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

# 1. Load and normalize CIFAR10 dataset
print("Load and normalize CIFAR10 dataset")
# Transform the output of torchvision datasets to tensors
transform: v2 = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size: int = 4

trainset: torchvision = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader: torch = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset: torchvision = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader: torch = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes: tuple = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img: torch.Tensor):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
dataiter: torch = iter(trainloader)
images, labels = next(dataiter);

# Print images
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
imshow(torchvision.utils.make_grid(images))


# 2. Define a CNN
print("Define a CNN, loss function and optimizer")
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(3, 6, 5)
        self.pool: nn.MaxPool2d = nn.MaxPool2d(2, 2)
        self.conv2: nn.Conv2d = nn.Conv2d(6, 16, 5)
        self.fc1: nn.Linear = nn.Linear(16 * 5 * 5, 120)
        self.fc2: nn.Linear = nn.Linear(120, 84)
        self.fc3: nn.Linear = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net: Net = Net()


# 3. Define a loss function and optimizer
criterion: nn = nn.CrossEntropyLoss()
optim: torch.optim = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# 4. Train the network
print("Train the network")
# Loop over the dataset multiple times
for epoch in range(2):

    running_loss: float = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Zero the parameter gradients
        optim.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save the trained model
PATH = './cifar_net.pt'
torch.save(net.state_dict(), PATH)


# 5. Test the network on the test data
print("Test the network on the test data")
dataiter: torch = iter(testloader)
images, labels = next(dataiter)

# Print images
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
imshow(torchvision.utils.make_grid(images))