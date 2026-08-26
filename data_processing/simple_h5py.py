import os
import h5py
import numpy as np

output_dir = "./"
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, "episode_1.hdf5")

print("=== 1. Creating Mock HDF5 File with Task Dataset ===")

N_FRAMES = 30
H, W, C = 480, 640, 3
ACTION_DIM = 7
TASK_STRING = "pick up the red cube"

head_imgs = np.random.randint(0, 256, (N_FRAMES, H, W, C), dtype=np.uint8)
wrist_imgs = np.random.randint(0, 256, (N_FRAMES, H, W, C), dtype=np.uint8)
states = np.random.randn(N_FRAMES, ACTION_DIM).astype(np.float32)
actions = np.random.randn(N_FRAMES, ACTION_DIM).astype(np.float32)

base_ts = np.linspace(0.0, 3.0, N_FRAMES)
ts_head = base_ts + np.random.uniform(-0.01, 0.01, N_FRAMES)
ts_wrist = base_ts + np.random.uniform(-0.01, 0.01, N_FRAMES)
ts_state = base_ts + np.random.uniform(-0.005, 0.005, N_FRAMES)
ts_action = base_ts + np.random.uniform(-0.005, 0.005, N_FRAMES)

# Define UTF-8 string type for HDF5 dataset compatibility
dt_str = h5py.string_dtype(encoding="utf-8")

with h5py.File(file_path, "w") as f:
    # Option A: Store root metadata attribute (existing approach)
    f.attrs["task"] = TASK_STRING
    f.attrs["fps"] = 10

    # # Option B1: Store as a scalar string dataset
    # f.create_dataset("task", data=TASK_STRING, dtype=dt_str)

    # # Option B2: Store as a per-frame string dataset (matching N_FRAMES length)
    # task_per_frame = np.array([TASK_STRING] * N_FRAMES, dtype=object)
    # f.create_dataset("task_per_frame", data=task_per_frame, dtype=dt_str)

    # Store observations and actions
    f.create_dataset(
        "observations/images/head",
        data=head_imgs,
        compression="gzip",
        chunks=True,
    )
    f.create_dataset(
        "observations/images/wrist",
        data=wrist_imgs,
        compression="gzip",
        chunks=True,
    )
    f.create_dataset("observations/state", data=states)
    f.create_dataset("action", data=actions)

    # Store timestamps
    f.create_dataset("timestamps/head_img", data=ts_head)
    f.create_dataset("timestamps/wrist_img", data=ts_wrist)
    f.create_dataset("timestamps/state", data=ts_state)
    f.create_dataset("timestamps/action", data=ts_action)

print(f"✅ File created: {file_path}\n")

# ================= 2. Read & Verify Stage =================
print("=== 2. Reading Task Data from Raw HDF5 ===")

with h5py.File(file_path, "r") as f:
    # Reading attribute
    task_from_attr = f.attrs["task"]

    # # Reading scalar dataset
    # task_from_ds = f["task"][()].decode("utf-8") if isinstance(f["task"][()], bytes) else str(f["task"][()])

    # # Reading per-frame dataset frame 0
    # task_frame_0 = f["task_per_frame"][0]
    # if isinstance(task_frame_0, bytes):
    #     task_frame_0 = task_frame_0.decode("utf-8")

    print(f"Read from f.attrs['task']          : {task_from_attr}") # Static global metadata
    # print(f"Read from f['task'][()]          : {task_from_ds}") # Single episode-level task stored as a dataset noted
    # print(f"Read from f['task_per_frame'][0] : {task_frame_0}") # Frame-by-frame dynamic task that change over time
    print("Top-level keys   :", list(f.keys()))
    print("Observation keys :", list(f["observations"].keys()))
    print("Timestamps keys  :", list(f["timestamps"].keys()))

    # Verify shapes
    print("\nDataset shapes:")
    print("  head image shape :", f["observations/images/head"].shape)
    print("  wrist image shape:", f["observations/images/wrist"].shape)
    print("  state shape      :", f["observations/state"].shape)
    print("  action shape     :", f["action"].shape)
    print("  head ts shape    :", f["timestamps/head_img"].shape)