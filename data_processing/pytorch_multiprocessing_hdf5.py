import torch
from torch.utils.data import Dataset
import h5py

class VLAEpisodeDataset(Dataset):
    # init() runs in the main process
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.file = None  # File handle initialized as None

    def __len__(self):# Lazy loading for multi-processing DataLoader
        self.__loadfile__()
        return self.file["episode_0/images"].shape[0]

    def __getitem__(self, idx):
        self.__loadfile__()

        image = self.file["episode_0/images"][idx]
        action = self.file["episode_0/actions"][idx]

        return torch.from_numpy(image), torch.from_numpy(action)


    # loadfile() runs in the child process
    def __loadfile__(self):
        # Lazy loading for multi-processing DataLoader
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, "r")


if __name__ == "__main__":
    dataset = VLAEpisodeDataset("./simple.hdf5")
    print(dataset[0])
    print(dataset.__len__())