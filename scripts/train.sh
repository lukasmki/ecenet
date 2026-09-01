#!/bin/sh
#SBATCH --account=m2834_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --job-name=ecetrain
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00

# Train an ECENet with the EVB mixture-of-experts read-out (ecenet/moe.py).
#
# Submit from the directory that holds logs/ — sbatch resolves --output before
# the job starts and will not create the directory:
#     mkdir -p logs && sbatch scripts/train.sh
#
# Re-submitting the same script RESUMES: the trainer restores model, optimiser,
# scheduler and best-so-far state from $CKPT when the file exists, so an 8 h
# wall clock is a checkpoint interval rather than a ceiling.

set -eu

cd $SCRATCH/ece/ecenet/
source venv/bin/activate

# ── Run configuration ───────────────────────────────────────────────────────
RUN=moe-evb-k4                        # names the checkpoint; bump it per experiment
DATA_TRAIN=data/train.xyz             # <-- set me: ASE-readable (extxyz, ...)
DATA_TEST=data/test.xyz               # <-- set me, or "" for no held-out test set
CKPT=checkpoints/${RUN}.mdl

mkdir -p "$(dirname "$CKPT")"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16             # CPU side is just tensorise + the data loop
export CUDA_VISIBLE_DEVICES=0         # single-process trainer: pin it to one GPU

echo "host=$(hostname)  job=${SLURM_JOB_ID}  run=${RUN}  ckpt=${CKPT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Config reaches Python through the environment so the heredoc stays quoted and
# nothing in the script body is shell-expanded.
export ECE_RUN="$RUN" ECE_TRAIN="$DATA_TRAIN" ECE_TEST="$DATA_TEST" ECE_CKPT="$CKPT"

python - <<'PY'
import os

import torch

from scripts.train_ecenet_moe import train_ecenet_moe

model, _, results = train_ecenet_moe(
    # ── Data ────────────────────────────────────────────────────────────
    train_path=os.environ['ECE_TRAIN'],
    test_path=os.environ['ECE_TEST'] or None,
    energy_key='energy',              # ASE info key holding the reference energy
    val_frac=0.1,

    # ── Mixture of experts ──────────────────────────────────────────────
    n_experts=4,                      # K diabatic experts over the shared trunk
    moe_mixture='evb',                # lowest eigenvalue of H = diag(V) + C
    moe_scope='atom',                 # one K x K problem per atom: size-consistent
    moe_coupling='mlp',               # learned C(R); 'const'/'none' are the ablations
    moe_coupling_topology='full',     # 'chain' gives a sparse tridiagonal H
    moe_coupling_init=0.05,           # eV/atom; nonzero keeps the E1-E0 gap open
    moe_expert_init=0.05,             # breaks the symmetry between experts at init
    # Left at 0: the collapse regulariser is a blunt instrument, and the
    # per-epoch "experts [...]" line in the log says whether it is needed. If
    # one expert drifts to ~1.0 and the rest to ~0, restart with 0.01.
    moe_diversity_weight=0.0,
    moe_diversity_kind='load',

    # ── Architecture ────────────────────────────────────────────────────
    r_cut_edge=5.0, r_cut_neighbor=4.0,
    l_max=3, n_max=4, embed_dim=32, n_layers=2, n_max_d=8,
    n_mp=2,                           # >= 2 turns on message passing

    # ── Optimisation ────────────────────────────────────────────────────
    n_epochs=400,
    batch_size=8,
    lr=1e-3,
    weight_decay=1e-5,
    lr_schedule='cosine',
    warmup_epochs=5,
    lr_min_factor=0.01,
    grad_clip=10.0,
    energy_weight=1.0,
    force_weight=10.0,                # forces carry the signal; energies anchor it
    stress_weight=0.0,                # raise if info['stress'] is present (eV/A^3)
    loss='mse',
    best_metric='weighted',

    # ── Precision / bookkeeping ─────────────────────────────────────────
    # float32 + TF32 is the A100 production setting; the float64 default is for
    # the finite-difference tests. A/B the val MAE against tf32=False once.
    dtype=torch.float32,
    tf32=True,
    eval_every=1,
    eval_batch_size=16,
    seed=42,
    checkpoint_path=os.environ['ECE_CKPT'],
    verbose=True,
)

print(f"\n[{os.environ['ECE_RUN']}] "
      f"val E={results['val_energy_mae']:.4f} eV/atom  "
      f"F={results['val_force_mae']:.4f} eV/A  |  "
      f"test E={results['test_energy_mae']:.4f}  F={results['test_force_mae']:.4f}  |  "
      f"{results['n_params']:,} params")

# Per-epoch expert usage is already in the log ("experts [...]"). For a
# per-structure breakdown of which expert owns which chemistry — mean weights
# and mean |C| per frame — run scripts/train_ecenet_moe.expert_report against
# this checkpoint afterwards.
assert model.mixture_head is not None
PY
