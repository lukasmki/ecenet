#!/bin/sh
#SBATCH --account=m2834_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --job-name=ecebase
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00

# Parameter-matched control for scripts/train.sh: a plain single-head ECENet
# with the SAME trunk and (to within a handful of weights) the same parameter
# count as the K=4 EVB mixture, so a difference between the two runs is about
# the *structure* of the read-out rather than its size.
#
# The extra capacity goes into the read-out MLP — the same place the mixture
# spends it. `matched_single_head` solves for the width; at the architecture
# below that is output_hidden_dims=[144] against the default [64].
#
# Submit both from the directory holding logs/, and keep every shared setting
# identical (data, split, seed, optimiser) or the comparison means nothing:
#     mkdir -p logs
#     sbatch scripts/train.sh
#     sbatch scripts/train_baseline.sh
#
# Re-submitting RESUMES from $CKPT, as in train.sh.

set -eu

cd $SCRATCH/ece/ecenet/
source venv/bin/activate

# ── Run configuration ───────────────────────────────────────────────────────
# DATA_TRAIN / DATA_TEST / val_frac / seed MUST match scripts/train.sh.
RUN=baseline-matched
DATA_TRAIN=data/train.xyz             # <-- set me: same file as scripts/train.sh
DATA_TEST=data/test.xyz               # <-- set me, or "" for no held-out test set
CKPT=checkpoints/${RUN}.mdl
REF_CKPT=checkpoints/moe-evb-k4.mdl   # the mixture run, for the drift check below

mkdir -p "$(dirname "$CKPT")"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0

echo "host=$(hostname)  job=${SLURM_JOB_ID}  run=${RUN}  ckpt=${CKPT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

export ECE_RUN="$RUN" ECE_TRAIN="$DATA_TRAIN" ECE_TEST="$DATA_TEST" ECE_CKPT="$CKPT"
export ECE_REF_CKPT="$REF_CKPT"

python - <<'PY'
import os

import torch

from scripts.train_ecenet_moe import matched_single_head
from scripts.train_ecenet_xyz import train_ecenet_xyz

# ── The trunk, shared verbatim with scripts/train.sh ────────────────────────
TRUNK = dict(
    r_cut_edge=5.0, r_cut_neighbor=4.0,
    l_max=3, n_max=4, embed_dim=32, n_layers=2, n_max_d=8,
    n_mp=2,                           # >= 2 turns on message passing
)
# The mixture this run is the control for. Only used to compute the target
# parameter count — none of it is built here.
MIXTURE = dict(n_experts=4, moe_mixture='evb', moe_scope='atom',
               moe_coupling='mlp', moe_coupling_topology='full')

# Drift guard: once the mixture run has checkpointed, its saved hparams are the
# authority on the trunk. Catches the two scripts silently diverging, which
# would invalidate the comparison without any visible symptom.
ref = os.environ.get('ECE_REF_CKPT', '')
if ref and os.path.exists(ref):
    hp = torch.load(ref, map_location='cpu', weights_only=False)['hparams']
    drift = {k: (v, hp[k]) for k, v in {**TRUNK, **MIXTURE}.items()
             if k in hp and hp[k] != v}
    if drift:
        raise SystemExit(f"[{os.environ['ECE_RUN']}] trunk disagrees with {ref} "
                         f"(here, there): {drift}")
    print(f"trunk matches {ref}")

# Solves for the read-out width; prints both parameter counts.
BASELINE = matched_single_head(**MIXTURE, **TRUNK)

model, _, results = train_ecenet_xyz(
    # ── Data (identical to scripts/train.sh) ────────────────────────────
    train_path=os.environ['ECE_TRAIN'],
    test_path=os.environ['ECE_TEST'] or None,
    energy_key='energy',
    val_frac=0.1,

    # ── Architecture: same trunk, single head, widened read-out ─────────
    **BASELINE,

    # ── Optimisation (identical to scripts/train.sh) ────────────────────
    n_epochs=400,
    batch_size=8,
    lr=1e-3,
    weight_decay=1e-5,
    lr_schedule='cosine',
    warmup_epochs=5,
    lr_min_factor=0.01,
    grad_clip=10.0,
    energy_weight=1.0,
    force_weight=10.0,
    stress_weight=0.0,
    loss='mse',
    best_metric='weighted',

    # ── Precision / bookkeeping (identical to scripts/train.sh) ─────────
    dtype=torch.float32,
    tf32=True,
    eval_every=1,
    eval_batch_size=16,
    seed=42,                          # same seed => same split as the mixture run
    checkpoint_path=os.environ['ECE_CKPT'],
    verbose=True,
)

print(f"\n[{os.environ['ECE_RUN']}] "
      f"val E={results['val_energy_mae']:.4f} eV/atom  "
      f"F={results['val_force_mae']:.4f} eV/A  |  "
      f"test E={results['test_energy_mae']:.4f}  F={results['test_force_mae']:.4f}  |  "
      f"{results['n_params']:,} params")

assert model.mixture_head is None, "this is the no-mixture control"
PY
