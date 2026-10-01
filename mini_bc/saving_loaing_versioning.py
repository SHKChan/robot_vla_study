from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

from pushT_bc import ActionStateSpecs, PushT_BC, evaluate, train


# Saving
def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch : int,
                    train_loss: float,
                    val_loss: float,
                    spec: ActionStateSpecs,
                    name: str | None = None) -> None:
    """Build and save a fresh checkpoint dict capturing the CURRENT weights.

    Building the dict once outside the training loop and re-saving it is a
    silent bug: the file's timestamp updates but its contents never change.

    Args:
        model: The nn.Module whose weights to save.
        optimizer: The optimizer whose state to save.
        epoch: The current epoch number.
        train_loss: Training loss at this epoch.
        val_loss: Validation loss at this epoch.
        spec: Model dimensions and normalization statistics.
        name: Output file path. Defaults to 'checkpoint-{epoch}.pth'.
    """
    torch.save({
        'model_state_dict': model.state_dict(), # learned params + buffers
        'optimizer_state_dict': optimizer.state_dict(), # Adam moments + step count
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': {
            # Needed to reconstruct the architecture before loading weight.
            **spec.to_dict(),
            'lr': 1e-4,
            'batch_size': 32,
        },
    }, name if name is not None else f'checkpoint-{epoch}.pth')


# Loading
def load_checkpoint(path: str, device: torch.device) -> dict:
    checkpoint: dict = torch.load(path,
                                  map_location=device, # needed on a CPU-only machine
                                  weights_only=False) # payload contains plain dicts

    cfg: dict = checkpoint['config']

    # Reconstruct the architecture FIRST -- load_state_dic matches by key name and shape into an existingmodel; It cannot infer the architecture.
    model: torch.nn.Module = PushT_BC(cfg['action_dim'], cfg['state_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    # --- Resuming training (not needed for inference) ---
    # Load model weight onto the target device BEFORE this line: the optimizer state tensor must end up on the same device as the parameters they track.
    optimizer: torch.optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch: int = checkpoint['epoch'] + 1

    return {
        'checkpoint': checkpoint,
        'model': model,
        'optimizer': optimizer,
        'epoch': start_epoch,
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
    }


# Versioning
def save_best_version(model: torch.nn.Module,
                      optimizer: torch.optim,
                      criterion: nn.MSELoss,
                      dataloader: DataLoader,
                      spec: ActionStateSpecs,
                      device: torch.device,
                      num_epochs: int = 50)-> None:
    best_val_loss: float = float('inf')

    for epoch in range(num_epochs):
        train_losses: list[float] = train(model, optimizer, criterion, dataloader, spec, device)
        train_loss: float = train_losses[-1]

        val_ret: dict[str, float] = evaluate(model, criterion, dataloader, spec, device)
        val_loss: float = val_ret['val_loss']

        if val_loss < best_val_loss:
            print(f"Found new best model with epoch: {epoch}, and val_loss: {val_loss}")
            best_val_loss = val_loss
            # Build the dict inside the branch so it captures the weights as they
            # are right now, at the new best epoch.
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, spec, 'mini_bc/best_model.pth')


def inference(ckpt:  dict, image: torch.Tensor, state: torch.Tensor, device: torch.device) -> torch.Tensor:
    model: torch.nn.Module = ckpt['model']
    model.eval()

    # Recover the exact statistics used during training
    norm: dict = ckpt['checkpoint']['config']
    a_mean: torch.Tensor = np.array(norm['action_norm']['mean'])
    a_std: torch.Tensor = np.array(norm['action_norm']['std'])
    s_mean: torch.Tensor = np.array(norm['state_norm']['mean'])
    s_std: torch.Tensor = np.array(norm['state_norm']['std'])

    with torch.no_grad():
        # Normalize the state on the way in...
        state_n: torch.Tensor = torch.as_tensor((state- s_mean) / s_std, dtype=torch.float32)
        # Unsqueeze(0) turns one sample into batch of one: [D] -> [1, D]
        action: torch.Tensor = model(image, state_n.unsqueeze(0).to(device))
        # ...and un-normalize the action on the way out, back to physical units
        action = action.cpu().numpy()[0] * a_std + a_mean

    return action

if __name__ == '__main__':
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the PushT dataset
    dataset: LeRobotDataset = LeRobotDataset("lerobot/pusht", video_backend="pyav")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,num_workers=4, pin_memory=True, drop_last=True)

    # #--- Normalization statistics--
    # # Must be standardized to roughly zero mean and unit variance
    # stats: dict = dataset.meta.stats
    # a_mean: torch.Tensor = torch.as_tensor(stats['action']['mean'], dtype=torch.float32, device=device)
    # # clamp_min guards against a dimension that never varies (std == 0-> inf).
    # a_std: torch.Tensor = torch.as_tensor(stats['action']['std'], dtype=torch.float32, device=device).clamp_min(1e-6)
    # s_mean: torch.Tensor = torch.as_tensor(stats['observation.state']['mean'], dtype=torch.float32, device=device)
    # s_std: torch.Tensor = torch.as_tensor(stats['observation.state']['std'], dtype=torch.float32, device=device).clamp_min(1e-6)

    # # Check data format
    # sample: dict = next(iter(dataloader))
    # print(sample.keys()) # dict_keys(['action', 'observation.state', 'observation.image', ...])

    # action_dim: int = sample["action"].shape[1]
    # state_dim: int = sample["observation.state"].shape[1]

    # spec: ActionStateSpecs = ActionStateSpecs(action_dim, state_dim, a_mean, a_std, s_mean, s_std)

    # # Training
    # model: PushT_BC = PushT_BC(action_dim, state_dim).to(device)
    # optimizer: torch.optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion: nn.MSELoss = nn.MSELoss()

    # save_best_version(model, optimizer, criterion, dataloader, spec, device, 5)

    # Loading
    checkpoint: dict = load_checkpoint('mini_bc/best_model.pth', device)

    # Check only the first (image, state) pair of the first batch.
    batch: dict = next(iter(dataloader))
    image: torch.Tensor = batch["observation.image"][0:1].to(device) # slice keeps the batch dim: [1, C, H, W]
    state: torch.Tensor = batch['observation.state'][0]              # raw [D]; inference() normalizes and unsqueezes
    action: torch.Tensor = batch['action'][0]                        # raw ground truth, for comparison

    inferred_action: torch.Tensor = inference(checkpoint, image, state, device)
    print(f"Ground truth action: {action}")
    print(f"Inferred action: {inferred_action}")