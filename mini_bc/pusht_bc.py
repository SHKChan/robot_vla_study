from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torchvision.models as models

# Load the PushT dataset
dataset: LeRobotDataset = LeRobotDataset("lerobot/pusht", video_backend="pyav")
dataloader: DataLoader = DataLoader(dataset, batch_size=32, shuffle=True)

# Check data format
sample: dict = next(iter(dataloader))
print(sample.keys()) # dict_keys(['action', 'observation.state', 'observation.image', ...])

# Mini BC network
class PushT_BC(nn.Module):
    def __init__(self):
        super().__init__()
        resnet: nn.Module = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.vision_encoder : nn.Module = resnet # output: [B, 512]
        self.fusion: nn.Module = nn.Sequential(
            nn.Linear(512+2, 128), # +2 for state
            nn.ReLU(),
            nn.Linear(128, 2), # PushT action is 2D
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        vis_feat: torch.Tensor = self.vision_encoder(image)
        combined: torch.Tensor = torch.cat([vis_feat, state], dim=1)
        action: torch.Tensor = self.fusion(combined)
        return action

# Training
model: PushT_BC = PushT_BC().cuda()
optimizer: torch.optim = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion: nn.MSELoss = nn.MSELoss()

for epoch in range(50):
    for batch in dataloader:
        image: torch.Tensor = batch["observation.image"].cuda(non_blocking=True)
        state: torch.Tensor = batch["observation.state"].cuda(non_blocking=True)
        action: torch.Tensor = batch["action"].cuda(non_blocking=True)

        # Forward
        pred_action: torch.Tensor = model(image, state)
        loss: torch.Tensor = criterion(pred_action, action)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}, Loss: {loss.item()}")