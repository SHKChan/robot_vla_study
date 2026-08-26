import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch


def show_single_image(img, title=None):
    plt.imshow(img)
    plt.axis('off')
    if(title):
        plt.title(title)
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

if __name__ == '__main__':
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # PyTorch Tensor [C,H,W] needs conversion
    img_tensor = torch.randn(3, 224, 224)
    img_np = img_tensor.permute(1, 2, 0).numpy()  # [C,H,W] → [H,W,C]
    show_single_image(img, 'Image NP')
    show_single_image(img_np, 'Image PyTorch to NP')

    actions = np.random.randn(100, 7)  # 100 frames, 7-dim action
    plot_action_trajectories(actions)

    images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(50)]
    episode_playback(images)