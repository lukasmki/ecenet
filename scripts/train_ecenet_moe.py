"""Training entry point for the EVB mixture-of-experts read-out, plus the
baseline comparison harness.

The architecture lives in ``ecenet/moe.py`` and is wired into ``ECENet`` behind
``n_experts > 1``; the optimisation loop is the small-dataset trainer
``scripts/train_ecenet_xyz.py``, which already carries the diversity
regulariser and the per-expert usage log. This module is the thin layer on top:
MoE-shaped defaults, and ``compare_mixtures`` — the reason it exists — which
trains the *same* architecture under each mixing rule so the only thing that
differs between runs is how the K expert energies are combined.

    from scripts.train_ecenet_moe import (train_ecenet_moe, compare_mixtures,
                                          matched_single_head)

    model, _, results = train_ecenet_moe(
        train_path='data/train-H2O_RPBE-D3.xyz',
        n_experts=4, moe_mixture='evb', n_epochs=200)

    table = compare_mixtures(train_path='data/train-H2O_RPBE-D3.xyz',
                             n_experts=4, n_epochs=200)

What the comparison controls for
--------------------------------
Every run shares the encoder, the expert heads, the data, the split and the
seed. 'single' is the plain one-head ECENet — the control that answers whether
K experts buy anything at all; 'mean' removes the gating; 'moe' is the ordinary
softmax-gated mixture; 'softmin' is the entropic smooth minimum; 'evb' is the
coupled Hamiltonian. Parameter counts are reported alongside the errors because
they are *not* equal — K expert heads plus K(K-1)/2 coupling heads cost more
read-out parameters than one head, and a fair reading of the table has to
account for that. ``matched_single_head`` builds the other control: a plain
model widened to the mixture's parameter count, so capacity is held fixed and
only the read-out's structure varies.

Notes on the defaults
---------------------
``moe_scope='atom'`` builds one K×K Hamiltonian per atom rather than one per
structure. That keeps the energy size-consistent and the couplings intensive;
``moe_scope='global'`` is the literal formulation and is available, but λ_min is
subadditive, so it does not decompose over non-interacting subsystems.

Staged training (§11 of the theory note) is available rather than automatic:
run stage 1 to specialise the experts — jointly, or one run per chemical regime
— then restart from that checkpoint with ``moe_freeze_experts=True`` to fit the
couplings against a now-fixed diabatic basis. Doing it in one call would have to
guess where the boundary between the stages belongs, which depends on the data.

``moe_diversity_weight`` defaults to 0. Expert collapse — one expert sitting
below all the others everywhere, driving c_0 → (1, 0, …, 0) — is the main
failure mode, but the load-balancing pressure is a blunt instrument, and
specialisation imposed through the *training data* (a run per chemical regime,
per §11's two-stage recipe) is usually the better tool. Turn it on after
watching the ``experts [...]`` usage line collapse, not before.
"""

import os
import sys  # repo root + scripts/ on path for `import ecenet` / the xyz trainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import torch
from train_ecenet_mptrj import print_flush
from train_ecenet_xyz import train_ecenet_xyz

# 'single' is not a mixing rule — it is the K=1 control, run through the same
# code path with the mixture head switched off entirely.
CONTROL = 'single'
DEFAULT_MIXTURES = (CONTROL, 'mean', 'softmin', 'moe', 'evb')


def train_ecenet_moe(
    train_path='train.xyz',
    test_path=None,
    n_experts=4,
    moe_mixture='evb',
    moe_scope='atom',
    moe_coupling='mlp',
    moe_coupling_topology='full',
    moe_diversity_weight=0.0,
    moe_diversity_kind='load',
    moe_freeze_experts=False,
    **kwargs,
):
    """Train an ECENet with a K-expert read-out on an ASE-readable dataset.

    Every keyword of ``train_ecenet_xyz`` is accepted and forwarded unchanged
    (geometry, architecture, optimiser, LES, checkpointing); the arguments named
    here are only surfaced because they are the ones a mixture run turns.

    ``n_experts=1`` degrades to the plain single-head model — useful for running
    the control through this same entry point.

    Returns ``(model, les_module, results)``, as the underlying trainer does.
    """
    return train_ecenet_xyz(
        train_path=train_path, test_path=test_path,
        n_experts=n_experts, moe_mixture=moe_mixture, moe_scope=moe_scope,
        moe_coupling=moe_coupling, moe_coupling_topology=moe_coupling_topology,
        moe_diversity_weight=moe_diversity_weight,
        moe_diversity_kind=moe_diversity_kind,
        moe_freeze_experts=moe_freeze_experts,
        **kwargs)


def compare_mixtures(mixtures=DEFAULT_MIXTURES, n_experts=4, seed=42,
                     verbose=True, **kwargs):
    """Train one model per mixing rule on identical data and report a table.

    ``mixtures`` may contain any of 'evb', 'moe', 'softmin', 'mean' and the
    K=1 control 'single'. Everything else — data, split, seed, architecture,
    optimiser — is shared, so a difference in the table is a difference between
    the mixing rules and nothing else.

    Returns ``{mixture: results}``, each entry the ``results`` dict of the
    underlying trainer (val/test energy and force errors, parameter count).
    """
    unknown = [m for m in mixtures if m not in DEFAULT_MIXTURES]
    if unknown:
        raise ValueError(f"unknown mixtures {unknown}; expected from {DEFAULT_MIXTURES}")
    kwargs.pop('n_experts', None)
    kwargs.pop('moe_mixture', None)

    table = {}
    for name in mixtures:
        if verbose:
            print_flush(f"\n{'=' * 72}\n=== {name} "
                        f"({'K=1 control' if name == CONTROL else f'K={n_experts}'})\n{'=' * 72}")
        # Same seed for every run: identical split, identical trunk init.
        run = dict(kwargs)
        if name == CONTROL:
            # The single-head control has no expert weights to balance; the
            # trainer rejects a nonzero diversity weight there, so drop it
            # rather than make the caller special-case the control.
            run['moe_diversity_weight'] = 0.0
        _, _, results = train_ecenet_moe(
            n_experts=1 if name == CONTROL else n_experts,
            moe_mixture='evb' if name == CONTROL else name,
            seed=seed, verbose=verbose, **run)
        table[name] = results

    if verbose:
        metric = 'mae' if 'val_energy_mae' in next(iter(table.values())) else 'rmse'
        print_flush(f"\n{'=' * 72}\nMixture comparison ({metric.upper()}; "
                    f"K={n_experts})\n{'=' * 72}")
        print_flush(f"{'mixture':<10} {'params':>9}  {'val E':>10} {'val F':>10} "
                    f"{'test E':>10} {'test F':>10}")
        for name, r in table.items():
            print_flush(f"{name:<10} {r['n_params']:>9,}  "
                        f"{r[f'val_energy_{metric}']:>10.5f} {r[f'val_force_{metric}']:>10.5f} "
                        f"{r[f'test_energy_{metric}']:>10.5f} {r[f'test_force_{metric}']:>10.5f}")
        print_flush("(energies eV/atom, forces eV/Å; parameter counts differ — "
                    "see the module docstring)")
    return table


def matched_single_head(n_experts, n_types=8, max_width=16384, verbose=True, **arch):
    """Architecture kwargs for a single-head ECENet with the mixture's parameter count.

    ``compare_mixtures`` deliberately does *not* equalise parameters — K expert
    heads plus K(K-1)/2 coupling heads cost more read-out weights than one head,
    and hiding that would be worse than reporting it. This closes the gap from
    the other side: given the mixture's architecture, it solves for the read-out
    hidden width that makes a plain ``n_experts=1`` model the same size, and
    returns the kwargs to build it.

        ARCH = dict(n_experts=4, l_max=3, n_max=4, embed_dim=32, n_layers=2,
                    n_max_d=8, n_mp=2)
        train_ecenet_moe(**ARCH, ...)                    # the mixture
        train_ecenet_moe(**matched_single_head(**ARCH), ...)   # the control

    The extra capacity is spent in the **read-out**, which is where the mixture
    spends it too — so a difference between the two runs is attributable to the
    *structure* of the read-out rather than to its size. Widening the trunk
    instead (``embed_dim``, ``n_layers``) answers a different and also
    interesting question: whether those parameters would have been better spent
    on the encoder. Neither control subsumes the other.

    ``n_types`` barely matters: it enters the two models identically through the
    trunk, and differs only in the mixture's per-(type, expert) and
    per-(type, pair) constants — order 100 parameters. The default probe value
    is fine unless K is large and the element set is huge.

    Note the counts will not match exactly: the read-out width is an integer, so
    the search returns the width whose count is closest to the target.
    """
    from ecenet import ECENet

    if n_experts < 2:
        raise ValueError(f"matched_single_head describes the control for a "
                         f"mixture model; got n_experts={n_experts}")
    arch = dict(arch)
    arch.pop('output_hidden_dims', None)          # this is what we solve for
    arch.pop('n_types', None)

    def count(**kw):
        return sum(p.numel() for p in ECENet(n_types=n_types, **kw).parameters())

    target = count(n_experts=n_experts, **arch)
    # params(width) is monotonically increasing, so bisect for the crossing and
    # then take whichever of the two neighbours lands closer.
    lo, hi = 1, max_width
    while lo < hi:
        mid = (lo + hi) // 2
        if count(n_experts=1, output_hidden_dims=[mid], **arch) < target:
            lo = mid + 1
        else:
            hi = mid
    width = min([w for w in (lo - 1, lo) if w >= 1],
                key=lambda w: abs(count(n_experts=1, output_hidden_dims=[w], **arch) - target))
    got = count(n_experts=1, output_hidden_dims=[width], **arch)
    # The mixture kwargs set the target but describe nothing about the control,
    # and are inert at K=1 — drop them so the returned dict is a plain model's
    # configuration rather than a mixture config with the head switched off.
    arch = {k: v for k, v in arch.items() if not k.startswith('moe_')}
    if verbose:
        print_flush(f"  parameter-matched control: output_hidden_dims=[{width}] "
                    f"→ {got:,} params vs the K={n_experts} mixture's {target:,} "
                    f"({100 * (got - target) / target:+.2f}%, n_types={n_types} probe)")
    return dict(arch, n_experts=1, output_hidden_dims=[width])


@torch.no_grad()
def expert_report(model, structures, type_map, e_ref, dtype=torch.float64,
                  device=None, stress_conv=1.0):
    """Which expert owns which structure — the interpretability read-out.

    Runs the model over already-loaded structure dicts (the format
    ``train_ecenet_xyz.tensorize`` takes) and returns, per structure, the mean
    expert weights and the mean coupling magnitude. A near-one-hot row means
    that configuration sits squarely inside one diabatic regime; a spread row
    with large |C| is a transition region, which is the regime EVB exists to
    represent smoothly.

    Returns ``(weights (n_struct, K), coupling (n_struct, P))``.
    """
    from train_ecenet_xyz import tensorize

    if model.mixture_head is None:
        raise ValueError("expert_report needs a model built with n_experts > 1")
    device = device or next(model.parameters()).device
    data = tensorize(structures, type_map, e_ref, model.r_cut_edge,
                     model.r_cut_neighbor, stress_conv, dtype, device)
    w_rows, c_rows = [], []
    for d in data:
        _, info = model.forward_pbc(
            d['pos'], d['types'], d['edge_i'], d['edge_j'], d['shift_e'],
            d['nb_src'], d['nb_dst'], d['shift_nb'], return_mixture=True)
        w_rows.append(info['weights'].mean(0))
        c_rows.append(info['coupling'].abs().mean(0))
    return torch.stack(w_rows), torch.stack(c_rows)


if __name__ == "__main__":
    train_ecenet_moe()
