from torch.utils.data import DataLoader

import torch
import torch.nn as nn

from pushT_bc import ActionStateSpecs

import gymnasium as gym, imageio, numpy as np, torch

class ENV:
    def __init__(self):
        pass

    def reset(self)->None:
        pass

    def extract_obs(self)->{torch.Tensor, torch.Tensor}:
        pass

    def step(self, action: torch.Tensor)->{torch.Tensor, torch.Tensor, bool, bool, dict[str, torch.Tensor]}:
        pass


def preprocess(img: torch.Tensor):
    pass


DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()                                  # no autograd graph → faster, less memory
def evaluate(model: nn.Module,
             spec: ActionStateSpecs,
             env: ENV,
             num_eval_episodes: int = 50) -> float:
    model.eval()
    success_count: int = 0

    for ep in range(num_eval_episodes):
        obs, _ = env.reset(seed=1000+ep) # seeded initial T pose
        for step in range(200):
            img, state = env.extract_obs(obs)   # 1. observe
            img_t = preprocess(img)             # 2. preprocess (same as training)
            state_t = (state - spec.state_mean) / spec.state_std    # normalize with Training stats
            action = model(img_t, state_t)      # 3. infer (no_grad)
            action = action * spec.action_std + spec.action_mean    # un-normalize
            obs, reward, terminated, truncated, info = env.step(action) # 4. execute

            # task solved
            if terminated:
                success_count += 1
                break
            # timeout
            if truncated:
                break

    success_rate: float  = success_count / num_eval_episodes

    return success_rate


# Adapted for gym-pusht
def to_tensor(pixels):                                   # MUST match training input
    x = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0   # HWC uint8 → CHW [0,1]
    return x.unsqueeze(0).to(DEVICE)

@torch.no_grad()
def rollout_eval(model, n_eps=50, max_steps=300, video_eps=3, a_mean=None, a_std=None, s_mean=None, s_std=None):
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    model.eval()
    records = []

    for ep in range(n_eps):
        obs, _ = env.reset(seed=1000 + ep)
        frames, max_cov, success = [], 0.0, False
        for t in range(max_steps):
            state = (torch.as_tensor(obs["agent_pos"], device=DEVICE).float() - s_mean) / s_std
            a = model(to_tensor(obs["pixels"]), state.unsqueeze(0))[0] * a_std + a_mean
            a = a.cpu().numpy().clip(0, 512)                 # stay inside the action space
            obs, reward, terminated, truncated, info = env.step(a)
            max_cov = max(max_cov, reward)
            if ep < video_eps: frames.append(env.render())
            if terminated: success = True; break
            if truncated: break
        records.append({"ep": ep, "success": success, "steps": t + 1, "max_cov": max_cov})
        # Recording video for debugging
        if frames: imageio.mimsave(f"eval_ep{ep}.mp4", frames, fps=10)
    return records