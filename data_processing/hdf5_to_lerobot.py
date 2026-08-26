import os
import shutil
from pathlib import Path
import h5py
import numpy as np
from data_processing.align_to_grid import align_to_grid
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def extract_task_instruction(f: h5py.File, default: str = "pick up the red cube") -> str | list[str]:
    """
    Extracts task instruction from HDF5 file.
    Order of precedence:
      1. 'task_per_frame' dataset (returns list of strings)
      2. 'task' dataset (returns single string)
      3. 'task' attribute (returns single string)
      4. Fallback default string
    """
    if "task_per_frame" in f:
        tasks = f["task_per_frame"][:]
        return [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in tasks]

    if "task" in f:
        raw_val = f["task"][()]
    else:
        raw_val = f.attrs.get("task", default)

    if isinstance(raw_val, (bytes, np.bytes_)):
        return raw_val.decode("utf-8")
    return str(raw_val)


def hdf5_to_lerobot(
    hdf5_episode_paths: list[str],
    repo_id: str,
    image_shape: tuple = (480, 640, 3),
    action_shape: tuple = (7,),
    fps: int = 10,
    tolerance_s: float = 1e-2,
    clean_existing: bool = True,
) -> LeRobotDataset:
    num_motors = action_shape[0]

    # Clean existing cached dataset directory to prevent FileExistsError
    if clean_existing:
        cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
        if cache_dir.exists():
            print(f"🧹 Removing existing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)

    # -------------------------------------------------------------------------
    # 1. Define LeRobot Dataset Schema
    # -------------------------------------------------------------------------
    features = {
        "observation.images.head": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": action_shape,
            "names": {"motors": [f"joint_{i}" for i in range(num_motors)]},
        },
        "action": {
            "dtype": "float32",
            "shape": action_shape,
            "names": {"motors": [f"joint_{i}" for i in range(num_motors)]},
        },
    }

    TARGET_FPS = fps

    # -------------------------------------------------------------------------
    # 2. Create Empty LeRobot Dataset Instance
    # -------------------------------------------------------------------------
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=TARGET_FPS,
        features=features,
        tolerance_s=tolerance_s,
    )

    # -------------------------------------------------------------------------
    # 3. Process HDF5 files and add frames
    # -------------------------------------------------------------------------
    for h5_path in sorted(hdf5_episode_paths):
        with h5py.File(h5_path, "r") as f:
            # Flexible task instruction reading
            task_data = extract_task_instruction(f)

            raw_timestamps = {
                "head_img": f["timestamps/head_img"][:],
                "wrist_img": f["timestamps/wrist_img"][:],
                "state": f["timestamps/state"][:],
                "action": f["timestamps/action"][:],
            }

            clean_grid, aligned_idx = align_to_grid(
                timestamps_dict=raw_timestamps,
                target_fps=TARGET_FPS,
                tolerance_s=0.05,
            )

            aligned_head_imgs = f["observations/images/head"][aligned_idx["head_img"]]
            aligned_wrist_imgs = f["observations/images/wrist"][aligned_idx["wrist_img"]]
            aligned_states = f["observations/state"][aligned_idx["state"]]
            aligned_actions = f["action"][aligned_idx["action"]]

            num_frames = len(clean_grid)

            # Handle per-frame task strings vs global episode task string
            if isinstance(task_data, list):
                aligned_tasks = [task_data[idx] for idx in aligned_idx["state"]]
            else:
                aligned_tasks = [task_data] * num_frames

            for i in range(num_frames):
                frame = {
                    "observation.images.head": aligned_head_imgs[i],
                    "observation.images.wrist": aligned_wrist_imgs[i],
                    "observation.state": aligned_states[i],
                    "action": aligned_actions[i],
                    "task": aligned_tasks[i],
                }
                dataset.add_frame(frame)

            print(f"✅ Processed {h5_path}, 💾 saving episode...")
            dataset.save_episode()

    print("✅ Successfully processed all episodes and converted to LeRobotDataset!"  )

    return dataset


if __name__ == "__main__":
    hdf5_episode_paths = ["./episode_0.hdf5", "./episode_1.hdf5"]

    dataset = hdf5_to_lerobot(
        hdf5_episode_paths=hdf5_episode_paths,
        repo_id="shkc/htr",
        tolerance_s=0.05,
    )

    print(dataset)