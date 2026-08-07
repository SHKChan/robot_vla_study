import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1: nn.Conv2d = nn.Conv2d(1, 6, 5)
        self.conv2: nn.Conv2d = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1: nn.Linear = nn.Linear(
            16*5*5, 120)  # 5*5 from image dimension
        self.fc2: nn.Linear = nn.Linear(120, 84)
        self.fc3: nn.Linear = nn.Linear(84, 10)

    def forward(self, input) -> torch.Tensor:
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1: torch.Tensor = F.relu(self.conv1(input))
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor
        s2: torch.Tensor = F.max_pool2d(c1, (2, 2))
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3: torch.Tensor = F.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4: torch.Tensor = F.max_pool2d(c3, 2)
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4: torch.Tensor = torch.flatten(s4, 1)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5: torch.Tensor = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6: torch.Tensor = F.relu(self.fc2(f5))
        # Fully connected layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output: torch.Tensor = self.fc3(f6)
        return output


net = Net()
print(net)

# All learned parameters of the model
params: list = list(net.parameters())
print(f"len(params): {len(params)}")
print(f"params[0].size(): {params[0].size()}")  # conv1's .weight

# Try a random input and forwardprop
input: torch.Tensor = torch.randn(1, 1, 32, 32)
out: torch.Tensor = net(input)
print(f"out: {out}")

# Loss function
target: torch.Tensor = torch.randn(10)
target = target.view(1, -1)  # make it the same shape as output
criterion: nn = nn.MSELoss()
loss: torch.Tensor = criterion(out, target)
print(f"loss: {loss}")
print(f"loss.grad_fn: {loss.grad_fn}") # MSELoss
print(f"loss.grad_fn.next_functions[0][0]: {loss.grad_fn.next_functions[0][0]}") # Linear
print(f"loss.grad_fn.next_functions[0][0].next_functions[0][0]: {loss.grad_fn.next_functions[0][0].next_functions[0][0]}") # ReLU

# # Zero gradients and backprops
# net.zero_grad()
# loss.backward()

# # Update the weight
# # Simplest update rule: weight = weight -(decrease) learning_rate(stride) * gradient(direction)
# learning_rate: float = 0.01
# for f in net.parameters():
#     with torch.no_grad():
#         f -= f.grad * learning_rate

# Or with optimizers to update the weights
optim: torch.optim = torch.optim.SGD(net.parameters(), lr=0.01)
optim.zero_grad()
loss.backward()
optim.step()