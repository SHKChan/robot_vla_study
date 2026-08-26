import numpy as np


def align_to_grid(
    timestamps_dict: dict[str, np.ndarray],
    target_fps: float,
    tolerance_s: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  """Aligns multi-modal timestamp arrays to a uniform target time grid using nearest-neighbor matching.

  Args:
      timestamps_dict (dict): Dictionary mapping modality names to 1D sorted
        timestamp arrays. Example: {'cam_head': ts_array, 'cam_wrist': ts_array,
        'state': ts_array}
      target_fps (float): Desired sampling frequency (Hz) for the output grid.
      tolerance_s (float, optional): Maximum allowed time gap (in seconds)
        between a target grid timestamp and an actual sensor timestamp. Defaults
        to half of the sampling interval (0.5 / target_fps).

  Returns:
      tuple[np.ndarray, dict]:
          - clean_grid (np.ndarray): Filtered uniform timestamp grid
            containing only valid time points.
          - aligned_indices (dict): Dictionary mapping each modality key to an
            array of indices corresponding to aligned raw data.
  """
  # Determine overlapping start and end time across all sensor modalities
  t_start = max(ts[0] for ts in timestamps_dict.values())
  t_end = min(ts[-1] for ts in timestamps_dict.values())

  dt = 1.0 / target_fps
  if tolerance_s is None:
    tolerance_s = dt * 0.5  # Default tolerance: half of sampling interval

  # Construct the uniform target time grid
  grid = np.arange(t_start, t_end, dt)
  aligned_indices = {}
  valid_mask = np.ones(len(grid), dtype=bool)

  for key, ts in timestamps_dict.items():
    # Perform binary search to find insertion indices
    idx = np.searchsorted(ts, grid)

    # Clip indices to safe range [1, len(ts) - 1] to avoid out-of-bounds errors
    idx = np.clip(idx, 1, len(ts) - 1)

    # Select nearest neighbor between left (idx-1) and right (idx) timestamps
    nearest_idx = np.where(
        np.abs(grid - ts[idx - 1]) < np.abs(grid - ts[idx]), idx - 1, idx
    )

    # Compute absolute time gap and update validity mask against tolerance
    gap = np.abs(grid - ts[nearest_idx])
    valid_mask &= gap <= tolerance_s

    # Store nearest neighbor indices for current modality
    aligned_indices[key] = nearest_idx

  # Filter grid and modality indices to keep only fully valid timestamp alignment
  clean_grid = grid[valid_mask]
  filtered_indices = {k: v[valid_mask] for k, v in aligned_indices.items()}

  print(f"🔛 Cleaned grid has {len(clean_grid)} valid time points.")
  return clean_grid, filtered_indices


def run_simple_test():
    # -------------------------------------------------------------------------
    # 1. Create mock timestamps with slight jitter and different sample rates
    # -------------------------------------------------------------------------
    # Target grid should be 10 Hz (dt = 0.1s)
    timestamps_dict = {
        # Camera 1 (~10Hz with small jitter)
        "cam_head": np.array([0.01, 0.11, 0.19, 0.32, 0.41, 0.51]),
        # Camera 2 (~10Hz with different jitter)
        "cam_wrist": np.array([0.00, 0.09, 0.21, 0.28, 0.39, 0.50]),
        # Robot State (20Hz higher sampling rate)
        "state": np.array([
            0.00,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
        ]),
    }

    target_fps = 10.0  # Target grid interval: 0.1s
    tolerance_s = 1 / (2 * target_fps ) # Maximum allowed timestamp difference: 50ms

    print("================ 1. Raw Timestamps ================")
    for key, ts in timestamps_dict.items():
        print(f"{key:10s} ({len(ts)} pts) : {ts}")

    # -------------------------------------------------------------------------
    # 2. Run alignment function
    # -------------------------------------------------------------------------
    clean_grid, aligned_indices = align_to_grid(
        timestamps_dict=timestamps_dict,
        target_fps=target_fps,
        tolerance_s=tolerance_s,
    )

    # -------------------------------------------------------------------------
    # 3. Print and verify alignment results
    # -------------------------------------------------------------------------
    print("\n================ 2. Aligned Results ================")
    print(f"Clean Target Grid ({len(clean_grid)} pts) : {clean_grid}")

    for key, idx_array in aligned_indices.items():
        # Retrieve actual timestamps using the returned indices
        matched_timestamps = timestamps_dict[key][idx_array]
        time_errors = np.abs(clean_grid - matched_timestamps)

        print(f"\nModality [{key}]:")
        print(f"  Aligned Indices    : {idx_array}")
        print(f"  Matched Timestamps : {matched_timestamps}")
        print(f"  Time Errors (s)    : {np.round(time_errors, 4)}")

        # Verification check
        assert len(idx_array) == len(
            clean_grid
        ), f"Length mismatch for {key}!"
        assert np.all(
            time_errors <= tolerance_s
        ), f"Tolerance exceeded for {key}!"

    print("\n✅ All alignment assertions passed successfully!")


if __name__ == "__main__":
    run_simple_test()