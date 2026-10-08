"""Run SmolVLA (base or fine-tuned) closed-loop in the PyBullet sim and measure success rate.

    perceive (top + wrist camera, joint state) -> SmolVLA -> chunk of 50 absolute joint targets
    -> execute the first K, then re-plan from the new observation (receding horizon)

Pipeline (see sim_env.py for the shared sim):
    1. python open_vla/record_demos.py --episodes 50          # scripted expert -> LeRobot dataset
    2. lerobot-train ... (command in record_demos.py)          # fine-tune smolvla_base
    3. python open_vla/smolvla_pybullet.py --policy-path outputs/smolvla_sim/checkpoints/last/pretrained_model

Run:
    python open_vla/smolvla_pybullet.py                       # base model (expected to fail)
    python open_vla/smolvla_pybullet.py --policy-path <ckpt>  # fine-tuned model
    python open_vla/smolvla_pybullet.py --expert              # scripted expert (sanity check, no model)
    add --no-gui --episodes 20 for a quick headless success rate
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sim_env import ArmSim, CTRL_HZ, TASK, expert_plan  # noqa: E402

BASE_MODEL = 'lerobot/smolvla_base'
BASE_STATS_DATASET = 'lerobot/svla_so100_pickplace'   # base model has no stats -> borrow a real SO-100 dataset's
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_policy(path: str):
    """Fine-tuned checkpoint: its saved processors already hold OUR dataset's stats.
    Base model: ships no state/action stats -> inject a real SO-100 dataset's (see smol_vla.py)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = DEVICE
    policy = SmolVLAPolicy.from_pretrained(path, config=cfg).to(DEVICE).eval()
    pre_over, post_over, base_stats = {'device_processor': {'device': DEVICE}}, {}, None
    if path == BASE_MODEL:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        base_stats = LeRobotDatasetMetadata(BASE_STATS_DATASET).stats
        feats = {**policy.config.input_features, **policy.config.output_features}
        nm = policy.config.normalization_mapping
        pre_over['normalizer_processor'] = {'stats': base_stats, 'features': feats, 'norm_map': nm}
        post_over['unnormalizer_processor'] = {'stats': base_stats,
                                               'features': policy.config.output_features, 'norm_map': nm}
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path,
                                         preprocessor_overrides=pre_over, postprocessor_overrides=post_over)
    return policy, pre, post, base_stats


def run_episode(env, args, policy=None, pre=None, post=None, init_state=None, rng=None) -> bool:
    env.reset(init_state_deg=init_state)
    if args.expert:
        plan = expert_plan(env, rng)
        chunks = iter([np.array(plan[i:i + args.exec_steps]) for i in range(0, len(plan), args.exec_steps)])
    else:
        policy.reset()

    t = 0
    while t < args.steps:
        # --- Perceive + Infer ---
        t0 = time.time()
        if args.expert:
            chunk = next(chunks, None)
            if chunk is None:
                break
        else:
            imgs, state = env.get_images(), env.get_state_deg()
            obs = {'observation.images.camera1': ArmSim.to_tensor(imgs['top']),
                   'observation.images.camera2': ArmSim.to_tensor(imgs['wrist']),
                   'observation.state': torch.tensor(state, dtype=torch.float32),
                   'task': TASK}
            with torch.inference_mode():
                chunk = post(policy.predict_action_chunk(pre(obs)))[0].float().cpu().numpy()   # [50, 6]
        if args.verbose:
            print(f'  t={t:4d}  infer {1000 * (time.time() - t0):5.0f} ms  '
                  f'state {env.get_state_deg().round(1)}  a0 {np.round(chunk[0], 1)}')

        # --- Execute first K actions, then re-plan ---
        for a in chunk[:args.exec_steps]:
            env.apply_action(a)
            t += 1
            if not args.no_gui:
                time.sleep(1.0 / CTRL_HZ)
            if env.success():
                return True
            if t >= args.steps:
                break
    for _ in range(15):                    # let a released cube settle before judging
        env.apply_action(env.target_deg)
    return env.success()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy-path', default=BASE_MODEL, help='hub id or local checkpoint dir')
    ap.add_argument('--expert', action='store_true', help='run the scripted expert instead of a model')
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--steps', type=int, default=300, help='max control steps per episode (30 Hz)')
    ap.add_argument('--exec-steps', type=int, default=10, help='actions executed per chunk before re-planning')
    ap.add_argument('--no-gui', action='store_true')
    ap.add_argument('--verbose', action='store_true', help='print every chunk')
    ap.add_argument('--seed', type=int, default=123, help='scene seed (differs from recording seed 0)')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    policy = pre = post = None
    grip_range, init_state = (0.0, 100.0), None          # our dataset's convention
    if not args.expert:
        print(f'Loading {args.policy_path} on {DEVICE}...')
        policy, pre, post, base_stats = load_policy(args.policy_path)
        if base_stats is not None:                        # base model: SO-100 units + start pose
            grip_range = (float(base_stats['action']['min'][5]), float(base_stats['action']['max'][5]))
            init_state = np.asarray(base_stats['observation.state']['mean'], dtype=np.float64)

    env = ArmSim(gui=not args.no_gui, grip_range=grip_range, seed=args.seed)
    wins = 0
    for ep in range(args.episodes):
        ok = run_episode(env, args, policy, pre, post, init_state, rng)
        wins += ok
        print(f'episode {ep + 1}/{args.episodes}: {"SUCCESS" if ok else "fail"}')
    print(f'\nSuccess rate: {wins}/{args.episodes} = {100 * wins / args.episodes:.0f}%')


if __name__ == '__main__':
    main()