"""SmolVLA inference on a REAL observation from a LeRobot dataset, compared to ground truth.

Instead of random images / zero state, we take one frame of a recorded SO-100
episode: its camera images, joint state and task string go in, and the
recorded next 50 actions are used as ground truth for the predicted chunk.

lerobot/smolvla_base (~450M params): up to 3 cameras + 6-D joint state in,
chunk of 50 x 6-D actions out. Fits a 4 GB GPU.

Setup (once, in the `vla` env):
    pip install "lerobot[smolvla,dataset]" av
    (video is decoded with PyAV by default; it bundles FFmpeg, so no torchcodec needed)

Run:
    python open_vla/smol_vla.py                                   # defaults below
    python open_vla/smol_vla.py --episode 3 --frame 120
    python open_vla/smol_vla.py --dataset <repo_id> --cams top,wrist

Note: smolvla_base is the PRETRAINED base, not fine-tuned on this dataset.
Expect actions in the right range and roughly the right direction, not an
exact match. Fine-tune on your own data before running it on a real arm.
"""
import argparse
import time

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


MODEL_ID: str = 'lerobot/smolvla_base'
DEFAULT_DATASET: str = 'lerobot/svla_so100_pickplace'   # any SO-100/101 LeRobot dataset works
DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_policy(model_id: str, device: str, stats: dict | None) -> tuple[SmolVLAPolicy, ...]:
    # The saved config says device='cuda'; override so it also runs on CPU.
    cfg: PreTrainedConfig = PreTrainedConfig.from_pretrained(model_id)
    cfg.device = device
    policy: SmolVLAPolicy = SmolVLAPolicy.from_pretrained(model_id, config=cfg)
    policy.to(device).eval()

    # The preprocessor normalizes state:  (x - mean) / std
    # the postprocessor un-normalizes actions: a * std + mean
    # smolvla_base is a multi-dataset BASE model: its saved processors carry no
    # mean/std for state/action, and LeRobot's (un)normalizer silently passes
    # tensors through when a key has no stats. So we inject the dataset's stats,
    # exactly like lerobot_train.py does when fine-tuning from smolvla_base.
    pre_over: dict = {'device_processor': {'device': device}}
    post_over: dict = {}
    if stats is not None:
        pre_over['normalizer_processor'] = {
            'stats': stats,
            'features': {**policy.config.input_features, **policy.config.output_features},
            'norm_map': policy.config.normalization_mapping,
        }
        post_over['unnormalizer_processor'] = {
            'stats': stats,
            'features': policy.config.output_features,
            'norm_map': policy.config.normalization_mapping,
        }
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=model_id,
        preprocessor_overrides=pre_over,
        postprocessor_overrides=post_over,
    )
    return policy, preprocess, postprocess


def load_frame(repo_id: str, episode: int, frame: int, chunk: int, video_backend: str) -> tuple[dict, LeRobotDatasetMetadata]:
    """Load one frame of one episode (downloads only that episode).

    delta_timestamps asks the dataset for the NEXT `chunk` actions too, so we get
    a ground-truth chunk [chunk, 6] to compare against the prediction.
    """
    meta = LeRobotDatasetMetadata(repo_id)
    delta = {'action': [i / meta.fps for i in range(chunk)]}
    ds = LeRobotDataset(repo_id, episodes=[episode], delta_timestamps=delta,
                        video_backend=video_backend)
    if not 0 <= frame < len(ds):
        raise SystemExit(f'--frame must be in 0..{len(ds) - 1} for episode {episode}')
    return ds[frame], meta


def build_observation(sample: dict, policy: SmolVLAPolicy, cam_order: list[str]) -> dict:
    """Map dataset keys -> the policy's expected keys.

    smolvla_base names its cameras camera1/2/3; datasets use names like
    'top', 'wrist'. We map them in `cam_order`. Fewer than 3 cameras is fine:
    SmolVLA just skips the missing ones.
    """
    policy_cams = [k for k in policy.config.input_features if k.startswith('observation.images.')]
    if len(cam_order) > len(policy_cams):
        raise SystemExit(f'Dataset has {len(cam_order)} cameras; policy accepts at most {len(policy_cams)}.')

    obs: dict = {}
    for policy_key, ds_key in zip(policy_cams, cam_order):
        obs[policy_key] = sample[ds_key]                   # [3, H, W] float in [0, 1]; resized to 512 inside
        print(f'  {ds_key:32s} -> {policy_key}  {tuple(sample[ds_key].shape)}')

    state = sample['observation.state']                    # [6] joint positions
    expected = policy.config.input_features['observation.state'].shape[0]
    if state.shape[0] != expected:
        raise SystemExit(f'State dim {state.shape[0]} != policy state dim {expected} (wrong robot?)')
    obs['observation.state'] = state
    obs['task'] = sample['task']                           # language instruction from the dataset
    return obs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=DEFAULT_DATASET)
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--frame', type=int, default=0, help='frame index within the episode')
    ap.add_argument('--cams', default=None,
                    help='comma-separated dataset camera names in camera1,2,3 order, e.g. "top,wrist"')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--video-backend', default='pyav', choices=['pyav', 'torchcodec'])
    ap.add_argument('--checkpoint-stats', action='store_true',
                    help="use only the checkpoint's own stats (reproduces the un-normalized output bug)")
    args = ap.parse_args()

    # Dataset metadata first: its mean/std are needed to build the processors.
    meta = LeRobotDatasetMetadata(args.dataset)
    stats = None if args.checkpoint_stats else meta.stats
    print(f'Loading {MODEL_ID} on {DEVICE} (normalization stats: '
          f'{"checkpoint only" if stats is None else args.dataset})...')
    policy, preprocess, postprocess = load_policy(MODEL_ID, DEVICE, stats)
    chunk_size: int = policy.config.chunk_size              # 50
    print(f'Loaded: {sum(p.numel() for p in policy.parameters()) / 1e6:.0f}M params')

    print(f'\nLoading {args.dataset}, episode {args.episode}, frame {args.frame}...')
    sample, meta = load_frame(args.dataset, args.episode, args.frame, chunk_size, args.video_backend)
    print(f'Dataset cameras: {meta.camera_keys}  fps: {meta.fps}')
    cam_order = ([f'observation.images.{c}' for c in args.cams.split(',')]
                 if args.cams else meta.camera_keys)
    obs = build_observation(sample, policy, cam_order)
    print(f'  task: "{obs["task"]}"')
    print(f'  state: {obs["observation.state"].numpy().round(1)}')

    # Flow matching starts from Gaussian noise -> seed it for repeatable output.
    torch.manual_seed(args.seed)
    policy.reset()
    batch: dict = preprocess(obs)

    # Sanity check: a normalized state should be roughly N(0, 1), not raw degrees.
    norm_state = batch['observation.state'].float().cpu()[0]
    print(f'  state after preprocess: {norm_state.numpy().round(2)}')
    if norm_state.abs().max() > 10:
        print('  WARNING: state is NOT normalized -> model sees out-of-distribution input, '
              'and actions will come out un-normalized (near 0).')

    # Warm-up pass: the first CUDA call includes kernel setup and is not real latency.
    policy.predict_action_chunk(batch)
    if DEVICE == 'cuda':
        torch.cuda.synchronize()

    start: float = time.time()
    pred = policy.predict_action_chunk(batch)               # [1, 50, 6], normalized
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    elapsed: float = time.time() - start
    pred = postprocess(pred).cpu()[0]                       # [50, 6], joint units
    print(f'\nPredicted chunk {tuple(pred.shape)} in {elapsed:.2f}s (after warm-up)')

    # ---- compare with the recorded actions ----
    gt = sample['action']                                   # [50, 6]
    valid = ~sample.get('action_is_pad', torch.zeros(chunk_size, dtype=torch.bool))  # end of episode -> padded
    err = (pred - gt).abs()[valid]                          # [n_valid, 6]
    names = meta.features['action'].get('names') or [f'j{i}' for i in range(gt.shape[1])]
    if isinstance(names, dict):                             # some datasets store {'motors': [...]}
        names = next(iter(names.values()))

    torch.set_printoptions(precision=1, sci_mode=False)
    print('\nFirst 3 steps   predicted:\n', pred[:3])
    print('First 3 steps   recorded :\n', gt[:3])
    print(f'\nMAE over {int(valid.sum())} valid steps (same units as the dataset, usually degrees):')
    for n, e in zip(names, err.mean(0)):
        print(f'  {n:16s} {e:6.2f}')


if __name__ == '__main__':
    main()