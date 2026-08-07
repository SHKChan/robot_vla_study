from typing import List

import torch
import numpy as np

# Tensor IndentationError
# 1. Directly from data
data: List[List[int]] = [[1, 2], [3, 4]]
x_data: torch.Tensor = torch.tensor(data)
# 2. From Numpy
np_array: np.ndarray = np.array(data)
x_np: torch.Tensor = torch.from_numpy(np_array)
# 3. From another tensor
# retains the properties of x_data
x_ones: torch.Tensor = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")
# overrides the datatype of x_data
x_rand: torch.Tensor = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")
# 4. With random or constant values
shape: tuple = (2, 3)
rand_tensor: torch.Tensor = torch.rand(shape)
ones_tensor: torch.Tensor = torch.ones(shape)
zeros_tensor: torch.Tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# Tensor Attributes
tensor: torch.Tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# Tensor Operations
# 1. We move our tensor to the GPU if available
device: torch.device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
tensor: torch.Tensor = tensor.to(device)
print(f"Device tensor is stored on: {tensor.device}")
# 2. Standard numpy-like indexing and slicing
tensor: torch.Tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(tensor)
# 3. Joining tensors
t1: torch.Tensor = torch.cat([tensor, tensor, tensor], dim=1)
print(f"Joined tensor: \n {t1} \n")
# 4. Multiplying tensors
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")
# 5. Matrix multiplication
# This computes the matrix product
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# 6. In-place operations, operations that have a suffix _
print(tensor, "\n")
tensor.add_(5)
print(tensor)


# Bridge with Numpy
# 1. Tensor to Numpy
t: torch.Tensor = torch.ones(5)
print(f"Tensor: {t}")
n: np.ndarray = t.numpy()
print(f"Numpy Array: {n}")
# A change in the numpy array reflects in the tensor
t.add_(1)
print(f"Tensor: {t}")
print(f"Numpy Array: {n}")

# 2. Numpy to Tensor
n: np.ndarray = np.ones(5)
t: torch.Tensor = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"Tensor: {t}")
print(f"Numpy Array: {n}")
