from typing import Iterator, TypedDict

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
from torch.utils.data import DataLoader

import torchvision.transforms as T

def episode_playback(images, from_tensor=False, title='Episode Playback'):
    if from_tensor:
        images = [img.permute(1, 2, 0).numpy() for img in images]

    fig, ax = plt.subplots()
    im = ax.imshow(images[0])
    ax.axis('off')
    plt.title(title)

    def update(frame):
        im.set_array(images[frame])
        return [im]

    anim = FuncAnimation(fig, update, frames=len(images), interval=50)
    # anim.save('episode.gif', writer='pillow')
    plt.show()

def plot_action_trajectories(actions, title='Action Trajectory'):
    actions_shape = actions.shape
    fig, axes = plt.subplots(actions_shape[1], 1, figsize=(10, 12))
    for i in range(actions_shape[1]):
        axes[i].plot(actions[:, i])
        axes[i].set_ylabel(f'Dim {i}')
    axes[-1].set_xlabel('Frame')
    plt.title(title)
    plt.tight_layout()
    # plt.savefig('actions.png', dpi=150)
    plt.show()

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
dataset: LeRobotDataset = LeRobotDataset("lerobot/pusht", video_backend="pyav")
dataloader: DataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
batch_iter: Iterator[LeRobotBatch] = iter(dataloader)
batch: LeRobotBatch = next(batch_iter)

# Image data augmentation
img_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])
eval_transform = T.Compose([
    T.Resize(224, antialias=True),
    T.CenterCrop(224),
])

imgs: torch.Tensor = batch['observation.image']
imgs_augmented: torch.Tensor = img_transform(imgs)
episode_playback(imgs, True, title='Image')
episode_playback(imgs_augmented, True, title='Image Augmentation')

# Actions and states normalization
scaler: MinMaxScaler = MinMaxScaler()
actions: torch.Tensor = batch['action']
action_mean: torch.Tensor = actions.mean(axis=0)
action_std: torch.Tensor = actions.std(axis=0).clip(min=1e-6)
# map targets to roughly zero mean, unit variance.
actions_norm: torch.Tensor = (actions - action_mean) / action_std

print(actions_norm.mean(0), actions_norm.std(0))   # ≈ [0, 0], [1, 1]
plot_action_trajectories(actions, title='Actions')
plot_action_trajectories(actions_norm, title='Normalized Actions')

# Inference: invert it, or the commands come out at the wrong scale.
# action_real = pred * action_std + action_mean

# # Min-max normalization to [-1, 1] also works:
# action_norm = 2 * (actions - min) / (max - min)- 1