# Server training

Run every command from the repository root. Use persistent storage for the
repository, especially `experiments/` (including its logs), `logs/`, and
`evaluation_results/`; do not rely on an ephemeral system disk.

## Create the Python environment

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Install CUDA-enabled PyTorch

First inspect the server driver with `nvidia-smi`. Then use the official
PyTorch installation instructions to select a CUDA-enabled wheel compatible
with that driver. Use a wheel compatible with PyTorch 2.9.1; the currently
verified code environment uses torch 2.9.1. Do not install a CPU-only torch
build for GPU training. The exact command intentionally is not hard-coded
because it depends on the server driver/CUDA compatibility.

After PyTorch is installed, install the remaining pinned dependencies:

```bash
pip install -r requirements.txt
```

The legacy `gym` package is not required; this repository imports Gymnasium.

## GPU preflight

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Do not start a full GPU run unless `torch.cuda.is_available()` prints `True`.

## Correctness preflight

```bash
python -m pytest tests/test_reward_correctness.py tests/test_joint_training_isolation.py tests/test_evaluation_correctness.py -q
python -m compileall core envs integration models heuristic
```

## Smoke training without changing source files

`config.TRAIN_CONFIG` contains `steps_per_cycle` and `total_cycles`. This
process-local override leaves `config.py` unchanged:

```bash
python -u -c "import config; config.TRAIN_CONFIG.update(steps_per_cycle=1, total_cycles=21); import train_dual; train_dual.main()"
```

`total_cycles=21` selects exactly one Phase 3 round because the training entry
uses `max(1, total_cycles - 20)`. Although the requested step count is one,
each PPO completes its indivisible rollout (2048 Nesting steps and 8192
Scheduling steps), so this is a functional smoke run rather than a one-step
unit test. It exercises Phase 1, Phase 2, Phase 3, and pair creation without
altering the formal defaults.

After success, the newest `experiments/exp_<timestamp>/models/` must contain:

```text
nesting_phase1_final.pt
scheduling_phase2.zip
nesting_joint_c1_final.pt
scheduling_joint_c1.zip
phase3_joint_c1.json
```

Validate the metadata and both referenced files:

```bash
python -c "import json,pathlib; e=max(pathlib.Path('experiments').glob('exp_*'),key=lambda p:p.stat().st_mtime); m=e/'models'/'phase3_joint_c1.json'; d=json.loads(m.read_text()); assert d['phase']=='phase3' and d['round']==1; assert all((m.parent/d[k]).is_file() for k in ('nesting_checkpoint','scheduling_checkpoint')); print(m)"
```

## Full training

The formal defaults remain in `config.py`. Start the complete Phase 1 → Phase
2 → Phase 3 run with:

```bash
python -u train_dual.py
```

For a long run, start a `tmux` session first so disconnecting SSH does not stop
training. Monitor free disk space and copy `experiments/` to durable storage.

## Local data

The untracked `data/` directory is not imported or read by the current
`train_dual.py` entry. It is not required on the server and must not be added
to Git merely to start training.
