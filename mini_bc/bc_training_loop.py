from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator, TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from bc_policy import BCPolicy

@dataclass(frozen=True)
class TrainConfig:
    repo_id: str = "lerobot/pusht"
    batch_size: int = 32
    num_epochs: int = 100
    lr: float = 1e-4
    num_workers: int = 4
    log_every: int = 10
    freeze_vision: bool = False
    checkpoint_path: Path = Path("bc_model_checkpoint.pth")

@dataclass(frozen=True)
class NormStats:
    """Per-dimension standardization statistics for one feature key."""
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_dataset(
        cls, dataset: LeRobotDataset, key: str, device: torch.device
    ) -> "NormStats":
        raw: dict[str, torch.Tensor] = dataset.meta.stats[key]
        mean = torch.as_tensor(raw["mean"], dtype=torch.float32, device=device)
        std = torch.as_tensor(raw["std"], dtype=torch.float32, device=device)
        return cls(mean=mean, std=std.clamp_min(1e-6))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

# Keys contain dots, so the functional TypedDict form is required.
LeRobotBatch = TypedDict(
    "LeRobotBatch",
    {
        "observation.image": torch.Tensor,   # [B, C, H, W] float32 in [0, 1]
        "observation.state": torch.Tensor,   # [B, S]
        "action": torch.Tensor,              # [B, A]
        "episode_index": torch.Tensor,       # [B]
        "frame_index": torch.Tensor,         # [B]
        "timestamp": torch.Tensor,           # [B]
        "index": torch.Tensor,               # [B]
        "task_index": torch.Tensor,          # [B]
    },
    total=False,   # 'task' (list[str]) and others may also be present
)

cfg: Final[TrainConfig] = TrainConfig()
# PushT dataset
dataset: LeRobotDataset = LeRobotDataset(cfg.repo_id, video_backend="pyav")
action_stats: NormStats = NormStats.from_dataset(dataset, "action", "cuda")
state_stats: NormStats = NormStats.from_dataset(
        dataset, "observation.state", "cuda"
    )
model: BCPolicy = BCPolicy(2).cuda()
optimizer: torch.optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
criterion: nn.MSELoss = nn.MSELoss()


# Frame-wise BC: shuffling across episodes is correct (i.i.d. state-action pairs).
dataloader: DataLoader[LeRobotBatch] = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=(torch.device.type == "cuda"),
    persistent_workers=cfg.num_workers > 0,
    drop_last=True,
)
num_batches: Final[int] = len(dataloader)
for epoch  in range(cfg.num_epochs):
    model.train()
    epoch_loss: float = 0.0

    batch_iter: Iterator[LeRobotBatch] = iter(dataloader)
    for batch in batch_iter:
        image: torch.Tensor = batch["observation.image"].cuda(non_blocking=True)
        state: torch.Tensor = state_stats.normalize(
            batch["observation.state"].cuda(non_blocking=True)
        )
        action: torch.Tensor = action_stats.normalize(
            batch["action"].cuda(non_blocking=True)
        )

        # Forward
        pred_action: torch.Tensor = model(image, state)
        loss: torch.Tensor = criterion(pred_action, action)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss: float = epoch_loss / len(dataloader)
    if epoch  % 10 == 0:
        print(f'Epoch: {epoch }, Loss: {avg_loss}')

# Save checkpoint
torch.save({
    'epoch': cfg.num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, 'bc_model_checkpoint.pth')


# Load checkpoint
checkpoint: dict = torch.load('bc_model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # switch to inference mode