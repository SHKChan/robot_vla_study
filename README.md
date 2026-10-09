
# VLA Study

## Environment Setup

Two separate conda environments, because **OpenVLA pins old library versions** (torch 2.2, transformers 4.40.1, timm 0.9.x) that conflict with LeRobot / SmolVLA.

| Env         | Purpose                                                            | Python | torch         |
| ----------- | ------------------------------------------------------------------ | ------ | ------------- |
| `vla`     | Course code, tiny VLA, LeRobot, SmolVLA, PyBullet sim, fine-tuning | 3.10   | 2.7.1 + cu126 |
| `openvla` | OpenVLA-7B code and inference                                      | 3.10   | 2.2.0 + cu121 |

## Prerequisites

- Miniconda installed (`conda init`, then restart the terminal)
- NVIDIA driver **560+** — check with `nvidia-smi`

---

## 1. `vla` — main study environment

```bash
conda create -n vla python=3.10 -y
conda activate vla

# torch first, pinned (cu126 works on both laptops, incl. older GPUs)
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu126

# LeRobot + SmolVLA + dataset tools, sim, video decoding, plotting
pip install "lerobot[smolvla,dataset]" pybullet av matplotlib
```

**Verify**

```bash
python -c "import torch, lerobot, pybullet; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.7.1+cu126 True`

**Notes**

- **Keep torch pinned.** If a later `pip install` upgrades torch, reinstall 2.7.1 with the command above.
- **Video backend: PyAV.** Use `--video-backend pyav` (scripts) and `--dataset.video_backend=pyav` (`lerobot-train`). This avoids torchcodec / system FFmpeg version mismatches.
- **Matplotlib "could not be resolved" in VS Code** → wrong interpreter selected, or install with `python -m pip install matplotlib` inside `vla`.

---

## 2. `openvla` — OpenVLA-7B

```bash
conda create -n openvla python=3.10 -y
conda activate openvla

pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/openvla/openvla.git
cd openvla && pip install -e .          # pins transformers / timm / tokenizers
pip install bitsandbytes pybullet        # 4-bit loading + sim

# optional — often fails to build on laptops; skip if it errors (slower but works)
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

**Hardware limit:** OpenVLA-7B needs ~15 GB GPU memory in bf16 (~7 GB in 4-bit).

| Laptop          | GPU memory | OpenVLA-7B locally? |
| --------------- | ---------- | ------------------- |
| RTX 3060 laptop | 6 GB       | ❌                  |
| Other laptop    | 4 GB       | ❌                  |

Use this env for **reading code and small CPU experiments**; run real inference on **Colab or a rented GPU**. SmolVLA (~450M params) in `vla` covers local hands-on work.

---

## Project layout

```
w6d4_vla/                  # Week 6 Day 4: action tokens (env: vla)
  action_tokenizer.py      # OpenVLA-style 256-bin action tokenizer
  tiny_vla.py              # tiny ViT + causal LM VLA, shape walkthrough
  train_tiny_vla.py        # train on a toy task, decode actions
open_vla/                  # SmolVLA + PyBullet (env: vla)
  smol_vla.py              # SmolVLA on a real LeRobot dataset frame vs ground truth
  sim_env.py               # shared sim: SO-100-style arm, cameras, scripted expert
  record_demos.py          # expert demos -> LeRobot dataset
  smolvla_pybullet.py      # run base / fine-tuned SmolVLA in sim, success rate
```

## Quick commands (`conda activate vla`)

```bash
# Week 6 Day 4
python w6d4_vla/action_tokenizer.py
python w6d4_vla/tiny_vla.py
python w6d4_vla/train_tiny_vla.py 2000

# SmolVLA on a real dataset frame
python open_vla/smol_vla.py --frame 150

# Sim sanity check (scripted expert, no model) — expect 100%
python open_vla/smolvla_pybullet.py --expert

# Record demos -> fine-tune -> evaluate
python open_vla/record_demos.py --episodes 50
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/sim_pickplace --dataset.root=data/sim_pickplace \
  --dataset.video_backend=pyav \
  --batch_size=8 --steps=20000 --save_freq=5000 \
  --output_dir=outputs/smolvla_sim --job_name=smolvla_sim
python open_vla/smolvla_pybullet.py \
  --policy-path outputs/smolvla_sim/checkpoints/last/pretrained_model --episodes 10
```

Out of GPU memory during training → `--batch_size=4`.

## VS Code

`Ctrl+Shift+P` → **Python: Select Interpreter**:

- `vla` for `w6d4_vla/` and `open_vla/`
- `openvla` only for OpenVLA-7B scripts
