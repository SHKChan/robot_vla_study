from dataclasses import dataclass

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torchvision.models as models

# Mini BC network
class PushT_BC(nn.Module):
    def __init__(self, action_dim=2, state_dim=2):
        super().__init__()
        resnet: nn.Module = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.vision_encoder : nn.Module = resnet # output: [B, 512]
        # ImageNet statistics, kept as buffers so they travel with the model.
        self.register_buffer('_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.fusion: nn.Module = nn.Sequential(
            nn.Linear(512+state_dim, 128), # vision features + raw state
            nn.ReLU(),
            nn.Linear(128, action_dim), # PushT action is 2-D
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image = (image - self._mean) / self._std
        vis_feat: torch.Tensor = self.vision_encoder(image)
        combined: torch.Tensor = torch.cat([vis_feat, state], dim=1)
        action: torch.Tensor = self.fusion(combined)
        return action


@dataclass
class ActionStateSpecs:
    """Everything needed to rebuild the model and map its I/O to physical units.

    Normalization statistics belong INSIDE the checkpoint. Kept in a separate
    file they drift out of sync, and without them the model's output cannot be
    converted back to physical units.
    """
    action_dim: int
    state_dim: int
    action_mean: torch.Tensor
    action_std: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor

    def to_dict(self):
        """Convert to plain Python types so the checkpoint loads without pickled arrays."""
        return {
            'action_dim': self.action_dim,
            'state_dim': self.state_dim,
            'action_norm': {'mean': self.action_mean.tolist(), 'std': self.action_std.tolist()},
            'state_norm': {'mean': self.state_mean.tolist(), 'std': self.state_std.tolist()},
        }


def train(model: nn.Module,
          optimizer: torch.optim,
          criterion: nn.MSELoss | nn.CrossEntropyLoss,
          dataloader: DataLoader,
          spec: ActionStateSpecs,
          device: torch.device,
          num_epochs: int = 1) -> list[float]:

    history: list[float] = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss: float = 0.0
        for batch in dataloader:
            image: torch.Tensor = batch["observation.image"].to(device, non_blocking=True)
            # Normalize inputs AND targets with the same statistics used later at inference time.
            state: torch.Tensor = (batch['observation.state'].to(device)- spec.state_mean) / spec.state_std
            action: torch.Tensor = (batch['action'].to(device)- spec.action_mean) / spec.action_std

            # Forward
            pred_action: torch.Tensor = model(image, state)
            loss: torch.Tensor = criterion(pred_action, action)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        mean_loss: float = epoch_loss / len(dataloader)
        history.append(mean_loss)

        # Report the epoch mean, not the last batch-- single batches are noisy.
        print(f'Epoch {epoch}: loss={history[-1]:.4f}')

    return history


@torch.no_grad()                                  # no autograd graph → faster, less memory
def evaluate(model: nn.Module,
             criterion: nn.MSELoss,
             dataloader: DataLoader,
             spec: ActionStateSpecs,
             device: torch.device) -> dict[str, float]:
    """Score the model on a dataloader without updating weights.

    Returns:
        dict: 'val_loss' (MSE in normalized units) and
              'val_px_err' (mean Euclidean action error in pixels).
    """
    model.eval()    # stored BatchNorm stats, dropout off
    total_loss, total_px_err, n = 0.0, 0.0, 0

    for batch in dataloader:
        image  = batch['observation.image'].to(device, non_blocking=True)
        state  = (batch['observation.state'].to(device) - spec.state_mean) / spec.state_std
        action = batch['action'].to(device)                      # raw pixels
        action_n = (action - spec.action_mean) / spec.action_std                       # same norm as training

        pred_n = model(image, state)
        B = action.shape[0]

        total_loss += criterion(pred_n, action_n).item() * B     # weight by batch size

        pred_px = pred_n * spec.action_std + spec.action_mean                        # un-normalize
        total_px_err += (pred_px - action).norm(dim=-1).sum().item()  # Euclidean px error
        n += B

    # An empty loader (e.g. a small val set with drop_last=True) would divide by zero.
    assert n > 0, "empty dataloader"

    ret: dict[str, float] = {'val_loss': total_loss / n,
                            'val_px_err': total_px_err / n}
    print(f"val_loss: {ret['val_loss']}, val_px_err: {ret['val_px_err']}")

    return ret


if __name__ == "__main__":
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the PushT dataset
    dataset: LeRobotDataset = LeRobotDataset("lerobot/pusht", video_backend="pyav")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,num_workers=4, pin_memory=True, drop_last=True)

    #--- Normalization statistics--
    # Must be standardized to roughly zero mean and unit variance
    stats: dict = dataset.meta.stats
    a_mean: torch.Tensor = torch.as_tensor(stats['action']['mean'], dtype=torch.float32, device=device)
    # clamp_min guards against a dimension that never varies (std == 0-> inf).
    a_std: torch.Tensor = torch.as_tensor(stats['action']['std'], dtype=torch.float32, device=device).clamp_min(1e-6)
    s_mean: torch.Tensor = torch.as_tensor(stats['observation.state']['mean'], dtype=torch.float32, device=device)
    s_std: torch.Tensor = torch.as_tensor(stats['observation.state']['std'], dtype=torch.float32, device=device).clamp_min(1e-6)

    # Check data format
    sample: dict = next(iter(dataloader))
    print(sample.keys()) # dict_keys(['action', 'observation.state', 'observation.image', ...])

    action_dim: int = sample["action"].shape[1]
    state_dim: int = sample["observation.state"].shape[1]

    spec: ActionStateSpecs = ActionStateSpecs(action_dim, state_dim, a_mean, a_std, s_mean, s_std)

    # Training
    model: PushT_BC = PushT_BC(action_dim, state_dim).to(device)
    optimizer: torch.optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion: nn.MSELoss = nn.MSELoss()

    train(model, optimizer, criterion, dataloader, spec, device)
    evaluate(model, criterion, dataloader, spec, device)