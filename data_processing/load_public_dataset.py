from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from data_visualization import episode_playback, plot_action_trajectories

# Load the PushT dataset (block-pushing task)
dataset = LeRobotDataset('lerobot/pusht', video_backend='pyav')

# Inspect dataset info
print(f'Length: {len(dataset)}')
print(f'Features: {dataset.features}')

# Get a single frame
frame = dataset[0]
# print(frame.keys())
# Example output: dict_keys(['action', 'observation.state', 'observation.image', 'episode_index', 'frame_index', 'timestamp'])

# Extract image and action
image = frame['observation.image']   # [C, H, W] tensor
action = frame['action']             # [D] tensor
state = frame['observation.state']   # [D] tensor
print('Image:', image)
print('Action:', action)
print('State:', state)

# Iterate in order over an episode
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in dataloader:
    images = batch['observation.image']  # [B, C, H, W]
    actions = batch['action']            # [B, D]
    states = batch['observation.state']  # [B, D]
    # data visualization
    episode_playback(images, True)
    print(states.shape)
    plot_action_trajectories(states)
    # training code...
    break