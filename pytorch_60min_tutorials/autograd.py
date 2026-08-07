import torch
from torchvision.models import resnet18, ResNet18_Weights

# Forward pass and Backward pass in PyTorch
# 1. Load model and data
model: torch.nn.Module = resnet18(weights=ResNet18_Weights.DEFAULT)
data: torch.Tensor = torch.rand(1, 3, 64, 64)  # input images
labels: torch.Tensor = torch.rand(1, 1000)  # ground truth

first_layer: torch.nn.Conv2d = model.conv1
print("--- First Layer (conv1) ---")
print("Weight Shape:", first_layer.weight.shape)
print("Weight Data (First 2 filters):\n", first_layer.weight[:1])
print("Bias Data:", first_layer.bias)

# 2. Forward pass
predictions: torch.Tensor = model(data)
# 3. Backward pass
loss: torch.Tensor = (predictions - labels).pow(2).sum()
loss.backward()
# 4. Optimizer
optim: torch.optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# 5. Update weights with gradient descent
optim.step()

first_layer = model.conv1
print("--- Updated First Layer (conv1) ---")
print("Weight Shape:", first_layer.weight.shape)
print("Weight Data (First 2 filters):\n", first_layer.weight[:2])
print("Bias Data:", first_layer.bias)


# Differentiation in Autograd
# Signals to Autograd that every operation on them should be tracked
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2
# Assume a and b to be learnable parameters, while Q to be the error
# L= v@Q = v1@Q1 + v2@Q2
external_grad: torch.Tensor = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
print(f"Expected gradients for a and b: {9*a**2}, {-2*b}")
print(f"Gradients for a and b: {a.grad}, {b.grad}")


# DAG tracks operations on all tensors
x: torch.Tensor = torch.rand(5, 5)
y: torch.Tensor = torch.rand(5, 5)
z: torch.Tensor = torch.rand((5, 5), requires_grad=True)

# The output tensor will require gradients even if only a single input has DAG enabled
a: torch.Tensor = x + y
print(f"Does `a` require gradients?: {a.requires_grad}")
b: torch.Tensor = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")


# Frozen parameters
model: torch.nn.Module = resnet18(weights=ResNet18_Weights.DEFAULT)
# Freeze all the parameters, only train the classifier
for param in model.parameters():
    param.requires_grad = False
# Replace the with a new classifier(unfrozen by default)
model.fc.weight.requires_grad = True
# Optimize only the classifier
optim: torch.optim = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
