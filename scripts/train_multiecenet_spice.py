"""Training script for MultiECENet (EVB) on SPICE, charged AND neutral species.

Why this exists separately from ``train_ecenet_spice.py``: that script trains a
plain ECENet on the neutral-only split (``train_large_neut_*.xyz``) and never
reads a total charge. Once anions and cations are in the training set, charge is
not optional information — acetate⁻ and a neutral radical can share a geometry
and an element list while differing by electron-volts, so a charge-blind model is
being asked to fit a one-to-many map and will average the two.

MultiECENet handles that by *sectoring*: each diabat carries a (charge,
multiplicity) label, and a structure of charge Q mixes only the diabats labelled
Q. Within a sector the diabats are resonance structures — carboxylate's two
equivalent C–O forms, guanidinium's three — and the EVB coupling lets the ground
state move between them smoothly. Across sectors nothing mixes, because a
resonance structure conserves total charge.

Data format: extended XYZ, as the neutral script, plus a total charge on the
comment line. Any of ``charge=`` / ``total_charge=`` / ``q=`` is accepted, and
``multiplicity=`` / ``spin=`` if present (SPICE is closed-shell throughout, so
multiplicity defaults to 1):

    26
    energy=-17081.3 charge=-1 multiplicity=1
    C  0.0 0.0 0.0  0.1 0.2 0.3
    ...

Usage:
    python scripts/train_multiecenet_spice.py                    # single process
    torchrun --nproc_per_node=4 scripts/train_multiecenet_spice.py   # DDP

Every option is a keyword argument of ``train_multiecenet_spice``.
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import math
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ecenet import MultiECENet


def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Fixed element → type mapping (10 elements in SPICE)
# ---------------------------------------------------------------------------

ELEMENT_TO_TYPE = {
    'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4,
    'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'I': 9,
}
N_TYPES = len(ELEMENT_TO_TYPE)
TYPE_NAMES = [k for k, v in sorted(ELEMENT_TO_TYPE.items(), key=lambda x: x[1])]

_ENERGY_RE = re.compile(r'(?<![A-Z_a-z])energy=([-+0-9.eE]+)')
# Charge/spin keys, longest first so 'total_charge' is not shadowed by 'charge'.
_CHARGE_RE = re.compile(r'(?<![A-Z_a-z])(?:total_charge|charge|q)=([-+0-9.eE]+)')
_SPIN_RE = re.compile(r'(?<![A-Z_a-z])(?:spin_multiplicity|multiplicity|spin)=([-+0-9.eE]+)')


class ChargeMissingError(ValueError):
    """Raised when a file carries no charge field at all.

    Deliberately fatal: defaulting a whole charged dataset to zero would train
    silently and wrongly, and the failure would only show up as a stubbornly
    high energy MAE much later.
    """


# ---------------------------------------------------------------------------
# Extended XYZ parser (charge-aware)
# ---------------------------------------------------------------------------

def parse_xyz_file(path, max_structures=None, dtype=np.float32, verbose=True,
                   require_charge=True, default_spin=1):
    """Parse an extended XYZ file into a list of structure dicts.

    Each dict contains positions (N,3) Å, forces (N,3) eV/Å, energy (eV),
    types (N,) int16, n_atoms, charge (int), spin (int multiplicity).

    ``require_charge=True`` raises ChargeMissingError if the file has no charge
    field anywhere, rather than quietly treating every structure as neutral.
    """
    structures = []
    t0 = time.time()
    unknown_elements = set()
    n_charge_found = 0

    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                n_atoms = int(line)
            except ValueError:
                continue

            comment = f.readline()
            m = _ENERGY_RE.search(comment)
            energy = float(m.group(1)) if m else 0.0
            mq = _CHARGE_RE.search(comment)
            if mq is not None:
                charge = int(round(float(mq.group(1))))
                n_charge_found += 1
            else:
                charge = 0
            ms = _SPIN_RE.search(comment)
            spin = int(round(float(ms.group(1)))) if ms else default_spin

            positions = np.empty((n_atoms, 3), dtype=dtype)
            forces = np.empty((n_atoms, 3), dtype=dtype)
            types = np.empty(n_atoms, dtype=np.int16)

            ok = True
            for i in range(n_atoms):
                parts = f.readline().split()
                elem = parts[0]
                if elem not in ELEMENT_TO_TYPE:
                    unknown_elements.add(elem)
                    ok = False
                    for _ in range(n_atoms - i - 1):
                        f.readline()
                    break
                types[i] = ELEMENT_TO_TYPE[elem]
                positions[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
                forces[i] = [float(parts[4]), float(parts[5]), float(parts[6])]

            if not ok:
                continue

            structures.append({
                'positions': positions, 'forces': forces, 'energy': energy,
                'types': types, 'n_atoms': n_atoms,
                'charge': charge, 'spin': spin,
            })

            if max_structures is not None and len(structures) >= max_structures:
                break
            if verbose and len(structures) % 50000 == 0 and len(structures) > 0:
                print_flush(f"  Parsed {len(structures):,} structures "
                            f"({time.time() - t0:.0f}s)...")

    if unknown_elements:
        print_flush(f"  Warning: skipped structures with unknown elements: {unknown_elements}")
    if structures and n_charge_found == 0:
        if require_charge:
            raise ChargeMissingError(
                f"{path}: no structure carries a charge field (looked for "
                f"charge= / total_charge= / q= on the comment line). Training a "
                f"MultiECENet on charge-unlabelled data would put every structure "
                f"in the q=0 sector, which is exactly the failure this model "
                f"exists to avoid. Pass require_charge=False to override if the "
                f"file really is all-neutral.")
        print_flush(f"  Warning: {path} has no charge field — treating all "
                    f"{len(structures):,} structures as neutral.")
    elif structures and n_charge_found < len(structures):
        print_flush(f"  Warning: {n_charge_found:,}/{len(structures):,} structures "
                    f"carry a charge field; the rest default to 0.")
    return structures


def sector_histogram(structures):
    """Counter over (charge, multiplicity) sectors present in the data."""
    return Counter((s['charge'], s['spin']) for s in structures)


def describe_sectors(structures, label, top=12):
    hist = sector_histogram(structures)
    total = sum(hist.values())
    print_flush(f"  {label}: {total:,} structures across {len(hist)} "
                f"(charge, multiplicity) sectors")
    for (q, m), n in sorted(hist.items(), key=lambda kv: -kv[1])[:top]:
        print_flush(f"      q={q:+d} M={m}:  {n:>9,}  ({100*n/total:5.2f}%)")
    if len(hist) > top:
        print_flush(f"      ... and {len(hist)-top} rarer sectors")
    return hist


# ---------------------------------------------------------------------------
# Per-element + per-sector energy reference
# ---------------------------------------------------------------------------

def compute_energy_reference(structures, sectors):
    """Least-squares reference: per-element energies PLUS a per-sector intercept.

    A neutral-atom linear reference cannot absorb the cost of adding or removing
    an electron, so on mixed-charge data the element-only fit leaves eV-scale
    residuals that differ systematically between sectors — the energy loss then
    spends its first epochs learning a constant offset per charge. Adding one
    indicator column per sector fits that offset directly, so the residuals the
    model actually trains on are comparable across sectors.

    Design matrix: [element counts | one-hot sector].

        E_i ≈ Σ_t n_it · e_ref[t]  +  q_ref[sector_i]

    Returns (e_ref (N_TYPES,), q_ref {sector: float}). lstsq's minimum-norm
    solution handles the built-in rank deficiency (element counts and the sector
    indicators are collinear whenever a sector has fixed composition).
    """
    sector_idx = {s: i for i, s in enumerate(sectors)}
    n, k = len(structures), len(sectors)
    A = np.zeros((n, N_TYPES + k), dtype=np.float64)
    E = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(structures):
        for t in s['types']:
            A[i, t] += 1
        A[i, N_TYPES + sector_idx[(s['charge'], s['spin'])]] = 1.0
        E[i] = s['energy']
    sol, _, rank, _ = np.linalg.lstsq(A, E, rcond=None)
    e_ref = sol[:N_TYPES]
    q_ref = {s: float(sol[N_TYPES + i]) for s, i in sector_idx.items()}
    return e_ref, q_ref, rank


def reference_energy(structure, e_ref, q_ref):
    return (sum(e_ref[t] for t in structure['types'])
            + q_ref.get((structure['charge'], structure['spin']), 0.0))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def to_device_tensors(structures, e_ref, q_ref, dtype, device):
    """Structure dicts → device tensors, with the reference energy subtracted.

    Charges and spins come back as plain int lists: they index the EVB sector,
    never enter an autograd graph, and MultiECENet takes them per-structure.
    """
    positions_list, forces_list, energies, types_list = [], [], [], []
    charges, spins = [], []
    for s in structures:
        positions_list.append(torch.tensor(s['positions'], dtype=dtype, device=device))
        forces_list.append(torch.tensor(s['forces'], dtype=dtype, device=device))
        types_list.append(torch.tensor(s['types'].astype(np.int64),
                                       dtype=torch.long, device=device))
        energies.append(torch.tensor(s['energy'] - reference_energy(s, e_ref, q_ref),
                                     dtype=dtype, device=device))
        charges.append(s['charge'])
        spins.append(s['spin'])
    return positions_list, forces_list, torch.stack(energies), types_list, charges, spins


def build_states(sectors, diabats_per_sector):
    """Expand {(charge, mult): count} into MultiECENet's flat ``states`` list.

    ``diabats_per_sector`` is an int (same for every sector) or a dict keyed by
    sector. Two is the useful default: one diabat per sector is a plain
    single-surface model for that charge (no coupling, nothing to mix), while
    two lets the EVB matrix describe a resonance pair.
    """
    states = []
    for sector in sorted(sectors):
        n = (diabats_per_sector.get(sector, 1)
             if isinstance(diabats_per_sector, dict) else int(diabats_per_sector))
        if n < 1:
            raise ValueError(f"sector {sector} needs at least one diabat, got {n}")
        states.extend([sector] * n)
    return states


class _MultiForwardWrapper(nn.Module):
    """Thin wrapper so DDP intercepts forward_batch_multi for gradient sync.

    Sector labels ride through as plain ints; they select which diabats mix and
    never touch the autograd graph.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, positions_list, types_list, charge=None, spin=None,
                topology=None, return_matrix=False):
        return self.model.forward_batch_multi(
            positions_list, types_list, charge=charge, spin=spin,
            topology=topology, return_matrix=return_matrix)


def make_batches(indices, n_atoms, batch_size, max_atoms_per_batch, rng,
                 bucket=False):
    """Shuffled batches, optionally size-bucketed under an atom budget.

    Bucketing sorts by atom count before packing so a batch holds similarly
    sized molecules — with per-structure topology that keeps the padding-free
    concatenated forward balanced, which matters most under DDP.
    """
    idx = list(indices)
    rng.shuffle(idx)
    if bucket:
        idx.sort(key=lambda i: n_atoms[i])
    batches, cur, cur_atoms = [], [], 0
    for i in idx:
        na = n_atoms[i]
        over_count = len(cur) >= batch_size
        over_atoms = (max_atoms_per_batch is not None
                      and cur and cur_atoms + na > max_atoms_per_batch)
        if over_count or over_atoms:
            batches.append(cur)
            cur, cur_atoms = [], 0
        cur.append(i)
        cur_atoms += na
    if cur:
        batches.append(cur)
    rng.shuffle(batches)     # bucketing must not make epoch order deterministic
    return batches


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_multiecenet_spice(
    train_xyz='train_charged_and_neutral.xyz',
    test_xyz='test_charged_and_neutral.xyz',
    # Data
    n_train=None,
    n_val=5000,
    n_test=None,
    require_charge=True,
    min_sector_count=1,        # drop sectors rarer than this (they cannot train)
    # EVB
    diabats_per_sector=2,      # int, or {(charge, mult): int}
    shared_trunk=True,
    mix_mode='eigvalsh',
    # Geometry / architecture (forwarded to every ECENet trunk)
    r_cut_edge=5.0,
    r_cut_neighbor=4.0,
    l_max=3,
    n_max=4,
    cutoff_type='cosine',
    embed_dim=32,
    n_layers=2,
    n_max_d=8,
    m_max=None,
    activation='silu',
    use_nonlinearity=True,
    n_grid=None,
    output_hidden_dims=None,
    analytic_ace_basis=True,
    bottleneck_dim=None,
    n_mp=1,
    mp_type='softmax',
    mp_dim=None,
    mp_n_heads=1,
    mp_msg_envelope=True,
    mp_l_attention=False,
    element_film=False,
    film_embed_dim=16,
    film_n_rbf=0,
    film_hidden=None,
    film_per_m=False,
    film_shift=False,
    # Optimiser
    lr=1e-3,
    weight_decay=1e-5,
    grad_clip=None,
    lr_schedule='plateau',
    scheduler_patience=10,
    warmup_epochs=0,
    lr_min_factor=0.0,
    early_stopping_patience=None,
    # Training
    n_epochs=100,
    batch_size=8,
    max_atoms_per_batch=None,
    bucket=False,
    energy_weight=1.0,
    force_weight=1.0,
    loss='mse',
    huber_delta=0.01,
    eval_metric='mae',
    eval_every=1,
    eval_max_samples=None,
    seed=42,
    dtype=torch.float64,
    precompute_topology=False,
    device=None,
    checkpoint_path='multiecenet_spice.mdl',
    verbose=True,
    # DDP (set by __main__ when torchrun is detected)
    rank=0,
    world_size=1,
    local_rank=0,
):
    is_ddp = world_size > 1
    is_main = (rank == 0)
    verbose = verbose and is_main
    if eval_metric not in ('mae', 'rmse'):
        raise ValueError(f"eval_metric must be 'mae' or 'rmse', got {eval_metric!r}")

    if device is None:
        device = (torch.device(f'cuda:{local_rank}') if torch.cuda.is_available()
                  else torch.device('cpu'))
    elif isinstance(device, str):
        device = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ── Data ────────────────────────────────────────────────────────────────
    if verbose:
        print_flush(f"Loading {train_xyz} ...")
    train_all = parse_xyz_file(train_xyz, max_structures=n_train,
                               verbose=verbose, require_charge=require_charge)
    if verbose:
        print_flush(f"Loading {test_xyz} ...")
    test_all = parse_xyz_file(test_xyz, max_structures=n_test,
                              verbose=verbose, require_charge=require_charge)

    if verbose:
        describe_sectors(train_all, "train")
        describe_sectors(test_all, "test")

    # Sectors the model will carry. A sector seen only a handful of times cannot
    # train a diabat, and silently keeping it would spend parameters on noise —
    # drop those structures instead, loudly.
    hist = sector_histogram(train_all)
    kept = {s for s, n in hist.items() if n >= min_sector_count}
    dropped = {s: n for s, n in hist.items() if s not in kept}
    if dropped and verbose:
        print_flush(f"  Dropping {sum(dropped.values()):,} structures in "
                    f"{len(dropped)} sector(s) below min_sector_count="
                    f"{min_sector_count}: {sorted(dropped)}")
    train_all = [s for s in train_all if (s['charge'], s['spin']) in kept]
    n_test_before = len(test_all)
    test_all = [s for s in test_all if (s['charge'], s['spin']) in kept]
    if verbose and len(test_all) < n_test_before:
        print_flush(f"  Dropping {n_test_before - len(test_all):,} test structures "
                    f"in sectors the model does not carry")
    if not train_all:
        raise ValueError("no training structures left after sector filtering")

    states = build_states(kept, diabats_per_sector)
    if verbose:
        print_flush(f"  EVB states ({len(states)} diabats over {len(kept)} sectors): "
                    f"{states}")

    # Split train/val by structure (SPICE conformers of one molecule are
    # correlated; a random split leaks, but it is what the neutral script does
    # and keeps the two comparable).
    perm = rng.permutation(len(train_all))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    train_structs = [train_all[i] for i in tr_idx]
    val_structs = [train_all[i] for i in val_idx]

    # ── Reference energies: per-element + per-sector intercept ──────────────
    e_ref, q_ref, rank_A = compute_energy_reference(train_structs, sorted(kept))
    if verbose:
        print_flush(f"  Reference fit rank {rank_A}/{N_TYPES + len(kept)}")
        print_flush("    per-element (eV): " + "  ".join(
            f"{TYPE_NAMES[t]}={e_ref[t]:.3f}" for t in range(N_TYPES)))
        print_flush("    per-sector intercept (eV): " + "  ".join(
            f"q={q:+d}/M={m}:{v:+.3f}" for (q, m), v in sorted(q_ref.items())))
        resid = np.array([s['energy'] - reference_energy(s, e_ref, q_ref)
                          for s in train_structs])
        print_flush(f"    residual energy: mean {resid.mean():+.4f} eV, "
                    f"std {resid.std():.4f} eV, |max| {np.abs(resid).max():.3f} eV")

    pos_train, frc_train, eng_train, typ_train, chg_train, spn_train = to_device_tensors(
        train_structs, e_ref, q_ref, dtype, device)
    pos_val, frc_val, eng_val, typ_val, chg_val, spn_val = to_device_tensors(
        val_structs, e_ref, q_ref, dtype, device)
    pos_test, frc_test, eng_test, typ_test, chg_test, spn_test = to_device_tensors(
        test_all, e_ref, q_ref, dtype, device)
    n_atoms_train = [p.shape[0] for p in pos_train]

    # ── Model ───────────────────────────────────────────────────────────────
    arch = dict(
        n_types=N_TYPES, r_cut_edge=r_cut_edge, r_cut_neighbor=r_cut_neighbor,
        l_max=l_max, n_max=n_max, embed_dim=embed_dim, n_layers=n_layers,
        n_max_d=n_max_d, m_max=m_max, n_grid=n_grid, cutoff_type=cutoff_type,
        activation=activation, use_nonlinearity=use_nonlinearity,
        output_hidden_dims=output_hidden_dims,
        analytic_ace_basis=analytic_ace_basis, bottleneck_dim=bottleneck_dim,
        n_mp=n_mp, mp_type=mp_type, mp_dim=mp_dim, mp_n_heads=mp_n_heads,
        mp_msg_envelope=mp_msg_envelope, mp_l_attention=mp_l_attention,
        element_film=element_film, film_embed_dim=film_embed_dim,
        film_n_rbf=film_n_rbf, film_hidden=film_hidden,
        film_per_m=film_per_m, film_shift=film_shift,
    )
    model = MultiECENet(states=states, shared_trunk=shared_trunk,
                        mix_mode=mix_mode, **arch)
    model = (model.double() if dtype == torch.float64 else model.float()).to(device)

    # Warm-start the per-state atomic baselines at the element reference's own
    # scale: the reference already removed the sector offsets, so zero is right,
    # but every diabat in a sector starts identical — the off-diagonal heads'
    # nonzero init is what breaks that symmetry (see MultiECENet.__init__).
    if verbose:
        n_par = sum(p.numel() for p in model.parameters())
        n_trunk = sum(p.numel() for p in model.trunks.parameters())
        n_head = sum(p.numel() for p in model.heads.parameters())
        print_flush(f"  MultiECENet: {n_par:,} params "
                    f"({len(model.trunks)} trunk(s) {n_trunk:,} + "
                    f"{len(model.heads)} heads {n_head:,}), sectors "
                    f"{sorted(model.sectors)}")

    raw_model = model
    train_fwd = _MultiForwardWrapper(model).to(device)
    if is_ddp:
        train_fwd = DDP(train_fwd,
                        device_ids=[local_rank] if device.type == 'cuda' else None,
                        find_unused_parameters=True)   # sectors idle per batch
    eval_fwd = _MultiForwardWrapper(raw_model)

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    if lr_schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=scheduler_patience)
    elif lr_schedule == 'cosine':
        scheduler = None      # stepped manually below (needs the warmup ramp)
    else:
        raise ValueError(f"unknown lr_schedule {lr_schedule!r}")

    topo_train = (raw_model.trunks[0].build_topology(pos_train)
                  if precompute_topology else None)

    # ── Loss ────────────────────────────────────────────────────────────────
    def _elem_loss(diff):
        if loss == 'mse':
            return diff ** 2
        if loss == 'l1':
            return diff.abs()
        abs_d = diff.abs()
        return torch.where(abs_d <= huber_delta, 0.5 * diff ** 2,
                           huber_delta * (abs_d - 0.5 * huber_delta))

    def _err(diff):
        return diff ** 2 if eval_metric == 'rmse' else diff.abs()

    def _final(total, count):
        if not count:
            return float('nan')
        m = total / count
        return math.sqrt(m) if eval_metric == 'rmse' else m

    # ── Evaluation, with per-sector breakdown and EVB diagnostics ───────────
    def evaluate(pos_l, frc_l, eng_t, typ_l, chg_l, spn_l, max_samples=None,
                 per_sector=False):
        """Returns (energy_metric, force_metric, sector_table, weight_stats).

        ``sector_table`` maps sector → (E metric, F metric, n) so a model that
        is fine on neutrals and broken on anions cannot hide inside a pooled
        average — which, on a set that is mostly neutral, it otherwise would.
        ``weight_stats`` reports how peaked the ground-state eigenvector is:
        mean max-weight near 1 means the diabats are barely mixing, which is
        both a modelling signal and the number that says whether top-k expert
        routing would pay off.
        """
        eval_fwd.eval()
        idx = list(range(len(pos_l)))
        if max_samples is not None and max_samples < len(idx):
            idx = list(rng.choice(idx, max_samples, replace=False))

        e_acc = f_acc = 0.0
        f_count = 0
        per_sec = {}
        w_max_acc, w_n = 0.0, 0
        for start in range(0, len(idx), 32):
            chunk = idx[start:start + 32]
            pos_b = [pos_l[i].detach().clone().requires_grad_(True) for i in chunk]
            typ_b = [typ_l[i] for i in chunk]
            q_b = [chg_l[i] for i in chunk]
            s_b = [spn_l[i] for i in chunk]
            with torch.enable_grad():
                eng_pred, H = eval_fwd(pos_b, typ_b, charge=q_b, spin=s_b,
                                       return_matrix=True)
                grads = torch.autograd.grad(eng_pred.sum(), pos_b, allow_unused=True)
            w = raw_model.ground_vector(H) ** 2
            w_max_acc += w.max(dim=-1).values.sum().item()
            w_n += len(chunk)

            for k, i in enumerate(chunk):
                n_at = pos_l[i].shape[0]
                de = _err((eng_pred[k] - eng_t[i]) / n_at).item()
                g = grads[k] if grads[k] is not None else torch.zeros_like(pos_b[k])
                df = _err(-g - frc_l[i]).sum().item()
                e_acc += de
                f_acc += df
                f_count += n_at * 3
                if per_sector:
                    key = (chg_l[i], spn_l[i])
                    acc = per_sec.setdefault(key, [0.0, 0.0, 0, 0])
                    acc[0] += de; acc[1] += df; acc[2] += n_at * 3; acc[3] += 1

        table = {k: (_final(v[0], v[3]), _final(v[1], v[2]), v[3])
                 for k, v in per_sec.items()}
        return (_final(e_acc, len(idx)), _final(f_acc, f_count), table,
                w_max_acc / max(w_n, 1))

    # ── Checkpoint ──────────────────────────────────────────────────────────
    best_val = float('inf')
    best_state = None
    best_test = (float('nan'), float('nan'))
    epochs_without_improvement = 0

    def save_checkpoint(epoch):
        if checkpoint_path is None or not is_main:
            return
        torch.save({
            'epoch': epoch,
            'model': raw_model.state_dict(),
            'best_state': best_state,
            'optimizer': optimizer.state_dict(),
            'best_val_weighted': best_val,
            'best_test_e_mae': best_test[0],
            'best_test_f_mae': best_test[1],
            'hparams': arch,
            # The 'evb' dict is what MultiECENetCalculator dispatches on and
            # rebuilds from; without it a checkpoint is indistinguishable from
            # a plain ECENet one.
            'evb': {
                'states': [list(s) for s in states],
                'shared_trunk': shared_trunk,
                'mix_mode': mix_mode,
            },
            'e_ref': e_ref,
            # Per-sector intercept: inference must add BOTH this and e_ref back,
            # so it is stored in the same self-describing spirit. A list of
            # [charge, multiplicity, eV] triples rather than a tuple-keyed dict,
            # so the checkpoint stays readable without unpickling custom keys.
            'sector_energy_reference': [[q, m, v] for (q, m), v in sorted(q_ref.items())],
            'element_to_type': ELEMENT_TO_TYPE,
            'energy_units': 'eV',
        }, checkpoint_path)

    # ── Training loop ───────────────────────────────────────────────────────
    if verbose:
        print_flush(f"\nTraining {n_epochs} epochs on {len(pos_train):,} structures "
                    f"({len(pos_val):,} val, {len(pos_test):,} test), device {device}")
    t_start = time.time()
    all_idx = list(range(len(pos_train)))

    for epoch in range(n_epochs):
        train_fwd.train()
        batches = make_batches(all_idx, n_atoms_train, batch_size,
                               max_atoms_per_batch, rng, bucket=bucket)
        if is_ddp:      # equal batch counts per rank, else DDP deadlocks
            n_b = len(batches) // world_size
            batches = batches[rank * n_b:(rank + 1) * n_b]
        if not batches:
            raise ValueError("no batches — lower batch_size or world_size")

        if lr_schedule == 'cosine':
            if epoch < warmup_epochs:
                lr_now = lr * (epoch + 1) / max(warmup_epochs, 1)
            else:
                prog = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
                lr_now = lr * (lr_min_factor + (1 - lr_min_factor)
                               * 0.5 * (1 + math.cos(math.pi * prog)))
            for gp in optimizer.param_groups:
                gp['lr'] = lr_now

        epoch_loss = 0.0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            pos_b = [pos_train[i].detach().clone().requires_grad_(True) for i in batch]
            typ_b = [typ_train[i] for i in batch]
            q_b = [chg_train[i] for i in batch]
            s_b = [spn_train[i] for i in batch]
            topo_b = [topo_train[i] for i in batch] if topo_train is not None else None

            eng_pred = train_fwd(pos_b, typ_b, charge=q_b, spin=s_b, topology=topo_b)
            eng_tgt = torch.stack([eng_train[i] for i in batch])
            n_at = torch.tensor([pos_train[i].shape[0] for i in batch],
                                dtype=dtype, device=device)
            energy_loss = _elem_loss((eng_pred - eng_tgt) / n_at).mean()

            if force_weight > 0:
                # allow_unused: a structure with no edges contributes only the
                # constant per-state baseline, so its position leaf never enters
                # the graph. Its forces are exactly zero.
                grads = torch.autograd.grad(eng_pred.sum(), pos_b,
                                            create_graph=True, allow_unused=True)
                force_loss = sum(
                    _elem_loss((-(g if g is not None else torch.zeros_like(pos_b[k])))
                               - frc_train[batch[k]]).mean()
                    for k, g in enumerate(grads)) / len(batch)
            else:
                force_loss = 0.0

            total_loss = energy_weight * energy_loss + force_weight * force_loss
            total_loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += total_loss.item()
        epoch_loss /= len(batches)

        # ── Eval ────────────────────────────────────────────────────────────
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            if is_ddp:
                dist.barrier()
            val_w_t = torch.tensor(float('inf'), device=device)
            if is_main:
                val_e, val_f, val_tab, w_max = evaluate(
                    pos_val, frc_val, eng_val, typ_val, chg_val, spn_val,
                    max_samples=eval_max_samples, per_sector=True)
                val_weighted = energy_weight * val_e + force_weight * val_f
                val_w_t = torch.tensor(val_weighted, device=device)
            if is_ddp:
                dist.broadcast(val_w_t, src=0)
            if scheduler is not None:
                scheduler.step(val_w_t.item())

            stop = torch.tensor(0, device=device)
            if is_main:
                improved = val_weighted < best_val
                if improved:
                    best_val = val_weighted
                    best_state = {k: v.detach().clone()
                                  for k, v in raw_model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if (early_stopping_patience is not None
                        and epochs_without_improvement >= early_stopping_patience):
                    stop = torch.tensor(1, device=device)
            if is_ddp:
                dist.broadcast(stop, src=0)

            if is_main:
                if improved:
                    best_test = evaluate(pos_test, frc_test, eng_test, typ_test,
                                         chg_test, spn_test,
                                         max_samples=eval_max_samples)[:2]
                save_checkpoint(epoch)
                if verbose:
                    print_flush(
                        f"epoch {epoch+1:>4d}/{n_epochs} loss {epoch_loss:.5f} | "
                        f"val E {val_e*1000:.2f} meV/atom F {val_f:.4f} eV/Å | "
                        f"test E {best_test[0]*1000:.2f} F {best_test[1]:.4f} | "
                        f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                        f"⟨max c²⟩ {w_max:.3f} | {time.time()-t_start:.0f}s")
                    # Per-sector table: the whole point of the charged run.
                    for sec in sorted(val_tab):
                        e_s, f_s, n_s = val_tab[sec]
                        print_flush(f"        q={sec[0]:+d} M={sec[1]} n={n_s:>6,}: "
                                    f"E {e_s*1000:8.2f} meV/atom   F {f_s:.4f} eV/Å")
            if stop.item() == 1:
                if verbose:
                    print_flush(f"Early stopping at epoch {epoch+1}")
                break

    if is_main and best_state is not None:
        raw_model.load_state_dict(best_state)
    if verbose:
        print_flush(f"\nDone in {time.time()-t_start:.0f}s. "
                    f"Best val [weighted] {best_val:.5f}; "
                    f"test E {best_test[0]*1000:.2f} meV/atom "
                    f"F {best_test[1]:.4f} eV/Å")
        print_flush(f"Checkpoint: {checkpoint_path}")
    return raw_model


if __name__ == '__main__':
    _rank = int(os.environ.get('RANK', 0))
    _world = int(os.environ.get('WORLD_SIZE', 1))
    _local = int(os.environ.get('LOCAL_RANK', 0))
    if _world > 1:
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo')
        if torch.cuda.is_available():
            torch.cuda.set_device(_local)

    train_multiecenet_spice(
        train_xyz='train_charged_and_neutral.xyz',
        test_xyz='test_charged_and_neutral.xyz',
        diabats_per_sector=2,
        n_epochs=200,
        batch_size=8,
        lr=1e-3,
        force_weight=10.0,
        rank=_rank, world_size=_world, local_rank=_local,
    )

    if _world > 1:
        dist.destroy_process_group()
