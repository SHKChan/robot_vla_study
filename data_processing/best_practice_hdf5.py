import h5py
import numpy as np

with h5py.File("./best_practice.hdf5", "w") as f:
    ep = f.create_group("episode_0")

    # Initialize dataset with unlimited capacity on axis 0 (maxshape set to None)
    img_ds = ep.create_dataset(
        "images",
        shape=(0, 224, 224, 3),
        maxshape=(None, 224, 224, 3),
        dtype=np.uint8,
        chunks=(1, 224, 224, 3),
        compression="gzip"
    )

    # Simulate data collection loop (frame-by-frame appending)
    for frame_idx in range(10):
        new_frame = np.random.randint(0, 256, (1, 224, 224, 3), dtype=np.uint8)

        # 1. Dynamically expand dataset size along axis 0
        img_ds.resize(img_ds.shape[0] + 1, axis=0)
        # 2. Write the latest frame
        img_ds[-1] = new_frame[0]