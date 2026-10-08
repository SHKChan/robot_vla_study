"""Record scripted-expert demos in the PyBullet sim as a LeRobot dataset (for fine-tuning SmolVLA).

Each control step (30 Hz) we save what the policy will later see and do:
    observation.images.camera1  top camera   [240, 320, 3]  (video)
    observation.images.camera2  wrist camera [240, 320, 3]  (video)
    observation.state           joint state  [6]  (deg, gripper 0..100)
    action                      expert target [6] (same units, absolute)
    task                        'Pick up the cube and place it in the box.'
Camera keys are named camera1/camera2 on purpose: they match smolvla_base's
input features, so no --rename_map is needed when fine-tuning.

Run (in the `vla` env, from the folder containing open_vla/):
    python open_vla/record_demos.py --episodes 50            # headless, ~5-10 min
    python open_vla/record_demos.py --episodes 3 --gui       # watch the expert

Then fine-tune (RTX 3060 6 GB: batch 8; 4 GB GPU: try batch 4):
    lerobot-train \\
      --policy.path=lerobot/smolvla_base \\
      --policy.push_to_hub=false \\
      --dataset.repo_id=local/sim_pickplace --dataset.root=data/sim_pickplace \\
      --dataset.video_backend=pyav \\
      --batch_size=8 --steps=20000 --save_freq=5000 \\
      --output_dir=outputs/smolvla_sim --job_name=smolvla_sim

And evaluate:
    python open_vla/smolvla_pybullet.py --policy-path outputs/smolvla_sim/checkpoints/last/pretrained_model --episodes 10
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sim_env import ArmSim, IMG_H, IMG_W, CTRL_HZ, JOINT_NAMES, TASK, expert_plan  # noqa: E402


def make_features():
    vid = {'dtype': 'video', 'shape': (IMG_H, IMG_W, 3), 'names': ['height', 'width', 'channels']}
    vec = {'dtype': 'float32', 'shape': (6,), 'names': JOINT_NAMES}
    return {
        'observation.images.camera1': vid,
        'observation.images.camera2': dict(vid),
        'observation.state': vec,
        'action': dict(vec),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--root', default='data/sim_pickplace')
    ap.add_argument('--repo-id', default='local/sim_pickplace')
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f'{root} exists; pass --overwrite to replace it.')
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=CTRL_HZ, features=make_features(),
                               root=root, robot_type='so100_sim', use_videos=True)

    env = ArmSim(gui=args.gui, grip_range=(0.0, 100.0), seed=args.seed)
    rng = np.random.default_rng(args.seed)
    kept, tried, t0 = 0, 0, time.time()
    while kept < args.episodes:
        tried += 1
        env.reset()
        plan = expert_plan(env, rng)
        plan += [plan[-1]] * 15                      # hold still at the end -> learns to stop
        for a in plan:
            imgs = env.get_images()
            ds.add_frame({
                'observation.images.camera1': imgs['top'],
                'observation.images.camera2': imgs['wrist'],
                'observation.state': env.get_state_deg().astype(np.float32),
                'action': np.asarray(a, dtype=np.float32),
                'task': TASK,
            })
            env.apply_action(a)
            if args.gui:
                time.sleep(1.0 / CTRL_HZ)
        if env.success():                             # keep only successful demos
            ds.save_episode()
            kept += 1
            print(f'episode {kept:3d}/{args.episodes}  frames {len(plan)}  ({time.time() - t0:.0f}s)')
        else:
            ds.clear_episode_buffer()
            print(f'  expert failed on attempt {tried}, discarded')
    ds.finalize()
    print(f'\nSaved {kept} episodes to {root.resolve()}')


if __name__ == '__main__':
    main()