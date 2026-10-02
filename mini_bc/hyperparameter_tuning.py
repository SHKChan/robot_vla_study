import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from pushT_bc import PushT_BC

def train()->None:
    pass

def evaluate()->None:
    pass

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: PushT_BC = PushT_BC(2, 2).to(device)
optimizer: torch.optim = torch.optim.Adam(model.parameters(), lr=1e-4)

# Learning-rate schedulers

# Cosine annealing: the LR follows half a cosine from its initial value down
# to eta_min over T_max steps. T_max should match your total epoch count.
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
for epoch in range(100):
    train(...)
    scheduler.step()    # Call ONCE per epoch, after the optimizer steps

# Or step decay: multiply the LR by gamma every step_size epochs.
scheduler = StepLR(optimizer, step_size=30, gamma=0.1) # x0.1 every 30 epochs


# Grid search vs random search
# Grid search (exhaustive)
lrs = [1e-3, 1e-4, 1e-5]
batch_sizes = [16, 32, 64]
for lr in lrs:
    for bs in batch_sizes:
        train(lr, bs)
        evaluate(lr, bs)

# Random search (sampled, more efficient)
for _ in range(10):
    lr = 10 ** np.random.uniform(-5,-3)
    bs = np.random.choice([16, 32, 64])
    train(lr, bs)
    evaluate(lr, bs)