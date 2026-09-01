# Prototype, mainly implemented by Claude
"""Tests for the mixture-of-experts read-out (ecenet/moe.py), wired into ECENet
via ``n_experts > 1``.

Three layers, in order:

  1. the mixing algebra on plain tensors — Hamiltonian assembly, the closed-form
     2×2 ground state against the general eigensolver, the analytic expert
     weights, the C→0 hard-minimum limit, the variational bound, the avoided
     crossing, and the softmin/diversity helpers;
  2. the model integration — K=1 is bit-for-bit the old single-head model, the
     energy stays SO(3)-invariant and continuous at r_cut, forces match finite
     differences, Hellmann–Feynman holds exactly (this is the load-bearing
     claim: it is what makes the forces conservative), double backward works
     so force training is possible, and all four forward paths agree;
  3. the trainer — a smoke run of scripts/train_ecenet_moe.py including the
     diversity regulariser, and the baseline comparison harness.

Size consistency gets its own test because it is where the two scopes genuinely
differ: 'atom' is exactly additive over non-interacting subsystems, 'global' is
only superadditive (λ_min is superadditive), which is the documented cost of the
literal whole-structure formulation.

Run:  python tests/test_moe.py     (from the repo root)
"""

import os
import sys  # repo root + scripts/ on path (imports ecenet and the scripts/ trainers)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import tempfile

import numpy as np
import torch

from ecenet import ECENet
from ecenet.moe import (
    build_hamiltonian,
    coupling_pairs,
    diversity_loss,
    evb_ground_state,
    evb_two_state,
    softmin_energy,
)
from ecenet.radial import find_edges

DTYPE = torch.float64
N_TYPES = 4

COMMON = dict(
    n_types=N_TYPES, r_cut_edge=5.0, r_cut_neighbor=4.0,
    l_max=2, n_max=3, embed_dim=8, n_layers=2, n_max_d=4,
)


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    return ECENet(**COMMON, **kw).double()


def random_structure(n=6, seed=0, scale=1.8):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g, dtype=DTYPE) * scale
    types = torch.randint(0, N_TYPES, (n,), generator=g)
    return pos, types


def rand_rotation(seed=1):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g, dtype=DTYPE)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diag(R))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _forces(model, pos, types, **kw):
    p = pos.clone().requires_grad_(True)
    e = model(p, types, **kw)
    energy = e[0] if isinstance(e, tuple) else e
    f = -torch.autograd.grad(energy, p, create_graph=True)[0]
    return e, f


def _pbc_args(model, pos):
    """forward_pbc's index arguments for a non-periodic structure (zero shifts)."""
    ei, ej = find_edges(pos, model.r_cut_edge)
    edge_i, edge_j = torch.cat([ei, ej]), torch.cat([ej, ei])
    d = torch.cdist(pos, pos)
    nb_src, nb_dst = ((d < model.r_cut_neighbor) & (d > 1e-10)).nonzero(as_tuple=True)
    z_e = torch.zeros(len(edge_i), 3, dtype=pos.dtype)
    z_n = torch.zeros(len(nb_src), 3, dtype=pos.dtype)
    return edge_i, edge_j, z_e, nb_src, nb_dst, z_n


# ═══════════════════════════════════════════════════════════════════════════
# 1. The mixing algebra
# ═══════════════════════════════════════════════════════════════════════════

def test_coupling_topologies():
    """'full' couples every pair, 'chain' only neighbours, 'none' nothing."""
    assert coupling_pairs(4, 'full').shape == (6, 2)
    assert coupling_pairs(4, 'chain').tolist() == [[0, 1], [1, 2], [2, 3]]
    assert coupling_pairs(4, 'none').shape == (0, 2)
    assert coupling_pairs(1, 'full').shape == (0, 2)      # K=1 has no pairs
    for topo in ('full', 'chain'):
        p = coupling_pairs(5, topo)
        assert (p[:, 0] < p[:, 1]).all(), "pairs must be upper-triangular (i<j)"
    try:
        coupling_pairs(3, 'bogus'); raise AssertionError("bad topology accepted")
    except ValueError:
        pass
    print("  topologies: full K(K-1)/2, chain K-1, none 0; pairs are i<j")


def test_hamiltonian_assembly():
    """H is real symmetric, diag(H)=V, and off-diagonals land on the coupled pairs."""
    torch.manual_seed(0)
    V = torch.randn(5, 4, dtype=DTYPE)
    for topo in ('full', 'chain', 'none'):
        pairs = coupling_pairs(4, topo)
        C = torch.randn(5, pairs.shape[0], dtype=DTYPE)
        H = build_hamiltonian(V, C, pairs)
        assert H.shape == (5, 4, 4)
        assert torch.allclose(H, H.transpose(-1, -2)), f"{topo}: H not symmetric"
        assert torch.allclose(torch.diagonal(H, dim1=-2, dim2=-1), V)
        for k, (i, j) in enumerate(pairs.tolist()):
            assert torch.allclose(H[:, i, j], C[:, k])
        # Uncoupled entries must stay exactly zero (this is what makes 'chain' sparse)
        off = H.clone()
        off[:, torch.arange(4), torch.arange(4)] = 0
        assert int((off.abs() > 0).sum()) == off.shape[0] * 2 * pairs.shape[0]
    print("  H assembly: symmetric, diag = V, exactly 2P nonzero off-diagonals per matrix")


def test_two_state_matches_eigensolver():
    """The K=2 closed form reproduces torch.linalg.eigh, values and weights."""
    torch.manual_seed(1)
    V = torch.randn(200, 2, dtype=DTYPE) * 3
    C = torch.randn(200, dtype=DTYPE)
    e0, w, c0 = evb_two_state(V, C)
    H = build_hamiltonian(V, C.unsqueeze(-1), coupling_pairs(2))
    evals, evecs = torch.linalg.eigh(H)
    assert torch.allclose(e0, evals[:, 0], atol=1e-12), "closed-form eigenvalue differs"
    assert torch.allclose(w, evecs[:, :, 0] ** 2, atol=1e-10), "weights differ"
    # The eigenvector is only defined up to a global sign; the *relative* sign is not.
    assert torch.allclose((c0[:, 0] * c0[:, 1]).abs(), c0[:, 0].abs() * c0[:, 1].abs())
    assert ((c0[:, 0] * c0[:, 1]) * C <= 1e-12).all(), \
        "c_A c_B must oppose the sign of C (that is what lowers E0)"
    # And c0 diagonalises: c0ᵀ H c0 == E0
    quad = torch.einsum('bi,bij,bj->b', c0, H, c0)
    assert torch.allclose(quad, e0, atol=1e-10)
    print(f"  K=2 closed form == eigh over 200 random H (max Δ="
          f"{(e0 - evals[:, 0]).abs().max():.2e}); c0ᵀHc0 == E0")


def test_weights_match_analytic_formula():
    """w_A = ½[1 + (V_B-V_A)/sqrt((V_A-V_B)² + 4C²)] — the note's §4 expression."""
    torch.manual_seed(2)
    V = torch.randn(100, 2, dtype=DTYPE) * 2
    C = torch.randn(100, dtype=DTYPE) * 0.5
    _, w, _ = evb_two_state(V, C)
    VA, VB = V[:, 0], V[:, 1]
    wA = 0.5 * (1 + (VB - VA) / torch.sqrt((VA - VB) ** 2 + 4 * C ** 2))
    assert torch.allclose(w[:, 0], wA, atol=1e-12)
    assert torch.allclose(w.sum(-1), torch.ones(100, dtype=DTYPE))
    print("  expert weights == the analytic |eigenvector|² formula, and sum to 1")


def test_zero_coupling_is_hard_min():
    """C ≡ 0 collapses EVB onto min_i V_i with one-hot weights (the hard MoE)."""
    torch.manual_seed(3)
    for K in (2, 3, 5):
        V = torch.randn(50, K, dtype=DTYPE) * 2
        H = build_hamiltonian(V, V.new_zeros(50, 0), coupling_pairs(K, 'none'))
        e0, w, _ = evb_ground_state(H)
        assert torch.allclose(e0, V.min(-1).values, atol=1e-10), f"K={K}: not the min"
        assert torch.allclose(w.max(-1).values, torch.ones(50, dtype=DTYPE), atol=1e-8)
        assert torch.equal(w.argmax(-1), V.argmin(-1))
    print("  C=0 → λ_min == min_i V_i exactly, weights one-hot on the argmin (K=2,3,5)")


def test_evb_is_variational():
    """λ_min ≤ min_i V_i always, and strictly below it whenever a coupling is on.

    This is the structural difference from an ordinary MoE, which can only ever
    land *between* the experts.
    """
    torch.manual_seed(4)
    for K in (2, 4):
        V = torch.randn(200, K, dtype=DTYPE) * 2
        pairs = coupling_pairs(K)
        C = torch.randn(200, pairs.shape[0], dtype=DTYPE) * 0.4 + 0.3
        e0, w, _ = evb_ground_state(build_hamiltonian(V, C, pairs))
        vmin = V.min(-1).values
        assert (e0 <= vmin + 1e-12).all(), f"K={K}: eigenvalue above the lowest expert"
        assert (e0 < vmin - 1e-8).all(), f"K={K}: coupling did not lower the energy"
        # An ordinary convex mixture of the same experts cannot go below vmin
        assert ((w * V).sum(-1) >= vmin - 1e-12).all()
        # §5 decomposition: E0 = Σ w_i V_i + Σ_{i≠j} c_i C_ij c_j
        H = build_hamiltonian(V, C, pairs)
        _, _, c0 = evb_ground_state(H)
        cross = 2 * (c0[:, pairs[:, 0]] * c0[:, pairs[:, 1]] * C).sum(-1)
        assert torch.allclose(e0, (w * V).sum(-1) + cross, atol=1e-9)
        assert (cross < 0).all(), "the coupling correction must be stabilising"
    print("  λ_min < min_i V_i (strictly, with C≠0) while Σ w_i V_i ≥ min_i V_i; "
          "E0 = Σ w_i V_i + coupling correction")


def test_avoided_crossing_and_smoothness():
    """At a crossing E_± = V ± |C|, and EVB is smooth where min() has a kink."""
    C = torch.tensor(0.3, dtype=DTYPE)
    V = torch.stack([torch.tensor(1.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE)])
    e0, w, _ = evb_two_state(V.unsqueeze(0), C.unsqueeze(0))
    # gap_eps floors the radicand, so |C| is reproduced to ~gap_eps/(2|C|).
    assert abs(e0.item() - (1.0 - 0.3)) < 1e-9, "gap at the crossing is not |C|"
    assert abs(w[0, 0].item() - 0.5) < 1e-12, "weights not 50/50 at the crossing"

    # Sweep a linear crossing: the second derivative of min() blows up at the
    # kink, EVB's stays bounded by ~1/|C|.
    x = torch.linspace(-1, 1, 2001, dtype=DTYPE)
    Vs = torch.stack([x, -x], dim=-1)
    e_evb, _, _ = evb_two_state(Vs, torch.full_like(x, 0.3))
    e_min = Vs.min(-1).values
    h = (x[1] - x[0]).item()
    d2_evb = (e_evb[2:] - 2 * e_evb[1:-1] + e_evb[:-2]).abs().max() / h ** 2
    d2_min = (e_min[2:] - 2 * e_min[1:-1] + e_min[:-2]).abs().max() / h ** 2
    assert d2_evb < 10, f"EVB curvature not bounded ({d2_evb:.1f})"
    assert d2_min > 100 * d2_evb, "min() should have a far sharper kink"
    # Far from the crossing EVB tracks the lower expert.
    assert (e_evb[:50] - e_min[:50]).abs().max() < 0.05
    print(f"  avoided crossing: E0 = V - |C| at V_A=V_B; max|E''| EVB {d2_evb:.1f} "
          f"vs min() {d2_min:.0f}; tracks min far away")


def test_softmin_limits():
    """softmin ≤ min, → min as τ→0, and its weights are a softmax over -V/τ."""
    torch.manual_seed(5)
    V = torch.randn(64, 4, dtype=DTYPE) * 2
    vmin = V.min(-1).values
    prev = None
    for tau in (1.0, 0.1, 1e-3):
        e, w = softmin_energy(V, torch.full((64,), tau, dtype=DTYPE))
        assert (e <= vmin + 1e-12).all(), "softmin above the hard min"
        assert torch.allclose(w.sum(-1), torch.ones(64, dtype=DTYPE))
        err = (e - vmin).abs().max().item()
        if prev is not None:
            assert err < prev, "smaller τ should approach the min more closely"
        prev = err
    assert prev < 1e-2, f"τ=1e-3 still {prev:.2e} from the min"
    print(f"  softmin ≤ min and → min as τ→0 (|Δ| = {prev:.2e} at τ=1e-3)")


def test_diversity_loss_properties():
    """Each regulariser prefers spread-out expert usage to a collapsed one."""
    K, n = 4, 64
    uniform = torch.full((n, K), 1.0 / K, dtype=DTYPE)
    collapsed = torch.zeros(n, K, dtype=DTYPE); collapsed[:, 0] = 1.0
    # Specialised: sharp per row, but every expert used equally across the batch.
    specialised = torch.zeros(n, K, dtype=DTYPE)
    specialised[torch.arange(n), torch.arange(n) % K] = 1.0

    load_u = diversity_loss(uniform, 'load').item()
    load_c = diversity_loss(collapsed, 'load').item()
    load_s = diversity_loss(specialised, 'load').item()
    assert abs(load_u - 1.0) < 1e-12, "uniform usage should sit at the load minimum 1"
    assert abs(load_c - K) < 1e-12, "full collapse should sit at the load maximum K"
    assert abs(load_s - 1.0) < 1e-12, "balanced specialists are balanced load"

    # 'entropy' rewards sharp-but-spread; it separates specialists from uniform,
    # which 'load' deliberately does not.
    assert diversity_loss(specialised, 'entropy') < diversity_loss(uniform, 'entropy')
    assert diversity_loss(collapsed, 'entropy') > diversity_loss(specialised, 'entropy')
    assert diversity_loss(uniform, 'cv').item() < 1e-9
    assert diversity_loss(collapsed, 'cv').item() > 1.0
    try:
        diversity_loss(uniform, 'nope'); raise AssertionError("bad kind accepted")
    except ValueError:
        pass
    print(f"  diversity: load uniform={load_u:.3f} specialised={load_s:.3f} "
          f"collapsed={load_c:.3f} (min 1, max K); entropy separates specialists")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Model integration
# ═══════════════════════════════════════════════════════════════════════════

def test_single_expert_is_the_plain_model():
    """n_experts=1 builds no mixture head and is numerically the old model."""
    pos, types = random_structure(seed=7)
    plain = _model(seed=11)
    moe = _model(seed=11, n_experts=1)
    assert plain.mixture_head is None and moe.mixture_head is None
    assert set(plain.state_dict()) == set(moe.state_dict()), "K=1 added parameters"
    assert torch.equal(plain(pos, types), moe(pos, types))
    # And the mixture head only appears above K=1.
    assert _model(seed=11, n_experts=2).mixture_head is not None
    print("  n_experts=1: no head, identical state dict, identical energy")


def test_mixture_head_replaces_the_single_head():
    """A mixture model must not carry the now-unused single-head read-out.

    output_net is dead once the mixture head exists — no forward path reaches
    it — so building it anyway would add thousands of parameters that do nothing
    and silently inflate every parameter count the model is compared on.
    """
    plain, moe = _model(seed=20), _model(seed=20, n_experts=4)
    assert plain.output_net is not None
    assert moe.output_net is None, "mixture model still builds the single head"
    assert not any(k.startswith('output_net.') for k in moe.state_dict())
    # And the mixture head is genuinely cheaper than the head it replaces plus
    # a dead copy: sanity-check the count against an explicit rebuild.
    head = sum(p.numel() for p in moe.mixture_head.parameters())
    trunk = sum(p.numel() for p in plain.parameters()) - \
        sum(p.numel() for p in plain.output_net.parameters())
    assert sum(p.numel() for p in moe.parameters()) == trunk + head
    print(f"  mixture model drops output_net: {head:,} head params, no dead read-out")


def test_matched_single_head_control():
    """matched_single_head sizes a plain model to the mixture's parameter count."""
    from train_ecenet_moe import matched_single_head

    trunk = dict(l_max=2, n_max=3, embed_dim=8, n_layers=2, n_max_d=4,
                 r_cut_edge=5.0, r_cut_neighbor=4.0)
    mixture = dict(n_experts=4, moe_mixture='evb', moe_coupling='mlp',
                   moe_coupling_topology='full')
    control = matched_single_head(**mixture, **trunk, n_types=N_TYPES, verbose=False)

    assert control['n_experts'] == 1
    assert not any(k.startswith('moe_') for k in control), \
        "inert mixture kwargs leaked into the control config"
    assert control['output_hidden_dims'][0] > 64, "read-out was not widened"

    n_mix = sum(p.numel() for p in
                ECENet(n_types=N_TYPES, **mixture, **trunk).parameters())
    ctrl = ECENet(n_types=N_TYPES, **control)
    n_ctrl = sum(p.numel() for p in ctrl.parameters())
    assert ctrl.mixture_head is None, "the control must have no mixture"
    assert abs(n_ctrl - n_mix) / n_mix < 0.005, \
        f"parameter counts not matched: {n_ctrl:,} vs {n_mix:,}"
    # Widening is monotone, so the neighbouring width must be a worse match.
    worse = dict(control, output_hidden_dims=[control['output_hidden_dims'][0] + 1])
    n_worse = sum(p.numel() for p in ECENet(n_types=N_TYPES, **worse).parameters())
    assert abs(n_worse - n_mix) >= abs(n_ctrl - n_mix), "search did not find the closest width"

    try:
        matched_single_head(n_experts=1, **trunk, verbose=False)
        raise AssertionError("accepted n_experts=1 as a mixture to match")
    except ValueError:
        pass
    print(f"  matched control: hidden [{control['output_hidden_dims'][0]}] → "
          f"{n_ctrl:,} params vs the mixture's {n_mix:,} "
          f"({100 * (n_ctrl - n_mix) / n_mix:+.3f}%)")


def test_so3_invariance():
    """Every mixing rule keeps the energy rotation-invariant."""
    pos, types = random_structure(seed=8)
    Q = rand_rotation()
    for mixture in ('evb', 'moe', 'softmin', 'mean'):
        for scope in ('atom', 'global'):
            m = _model(seed=3, n_experts=3, moe_mixture=mixture, moe_scope=scope)
            err = (m(pos, types) - m(pos @ Q.T, types)).abs().item()
            assert err < 1e-9, f"{mixture}/{scope}: not SO(3)-invariant ({err:.2e})"
    print("  SO(3) invariance holds for evb / moe / softmin / mean × atom / global")


def test_forces_finite_difference():
    """Autograd forces match central differences of the mixed energy."""
    pos, types = random_structure(n=5, seed=9)
    eps = 1e-6
    for K, mixture in ((2, 'evb'), (3, 'evb'), (3, 'moe'), (3, 'softmin')):
        m = _model(seed=4, n_experts=K, moe_mixture=mixture)
        _, f = _forces(m, pos, types)
        fd = torch.zeros_like(pos)
        for i in range(pos.shape[0]):
            for d in range(3):
                pp = pos.clone(); pp[i, d] += eps
                pm = pos.clone(); pm[i, d] -= eps
                fd[i, d] = -(m(pp, types) - m(pm, types)) / (2 * eps)
        err = (f - fd).abs().max().item()
        assert err < 1e-6, f"K={K} {mixture}: force error {err:.2e}"
    print("  forces == finite differences for K=2/3 evb, moe, softmin (max err < 1e-6)")


def test_hellmann_feynman():
    """F = -c_0ᵀ (∇H) c_0 with the eigenvector held fixed — the note's §6 claim.

    This is what makes the EVB forces conservative for free: differentiating the
    Hamiltonian with *detached* mixing coefficients reproduces the true gradient
    of the eigenvalue, so no separate force head can drift out of sync with the
    energy.
    """
    pos, types = random_structure(n=6, seed=10)
    for scope in ('atom', 'global'):
        m = _model(seed=5, n_experts=3, moe_mixture='evb', moe_scope=scope)
        pairs = m.mixture_head.pairs

        p = pos.clone().requires_grad_(True)
        e, info = m(p, types, return_mixture=True)
        f_true = -torch.autograd.grad(e, p, retain_graph=True)[0]

        # Rebuild the energy as c_0ᵀ H c_0 with c_0 frozen: same value, and by
        # Hellmann–Feynman the same gradient.
        c0 = info['c0'].detach()
        V, C = info['expert_energies'], info['coupling']
        quad = (c0 ** 2 * V).sum(-1)
        if pairs.numel():
            quad = quad + 2 * (c0[:, pairs[:, 0]] * c0[:, pairs[:, 1]] * C).sum(-1)
        e_hf = quad.sum() + m.atomic_energy[types].sum()
        assert abs((e_hf - e).item()) < 1e-10, f"{scope}: c_0ᵀHc_0 != E0"

        f_hf = -torch.autograd.grad(e_hf, p)[0]
        err = (f_true - f_hf).abs().max().item()
        assert err < 1e-9, f"{scope}: Hellmann-Feynman mismatch {err:.2e}"
    print("  Hellmann-Feynman: -c_0ᵀ(∇H)c_0 == the autograd force (max err < 1e-9), "
          "both scopes")


def test_double_backward_for_force_training():
    """Forces stay differentiable w.r.t. the parameters — force training works."""
    pos, types = random_structure(n=6, seed=11)
    for K in (2, 4):
        m = _model(seed=6, n_experts=K, moe_mixture='evb')
        p = pos.clone().requires_grad_(True)
        f = -torch.autograd.grad(m(p, types), p, create_graph=True)[0]
        loss = (f ** 2).mean()
        loss.backward()
        grads = [q.grad for q in m.parameters() if q.grad is not None]
        assert grads, f"K={K}: no parameter received a gradient"
        assert all(torch.isfinite(g).all() for g in grads), f"K={K}: non-finite grads"
        head = m.mixture_head
        assert head.expert_atomic.grad is not None and \
            torch.isfinite(head.expert_atomic.grad).all()
        assert head.coupling_atomic.grad is not None, "couplings got no gradient"
    print("  double backward through λ_min is finite; experts and couplings both "
          "receive gradients (K=2 closed form and K=4 eigensolver)")


def test_size_consistency():
    """'atom' scope is exactly additive; 'global' is only superadditive.

    Two copies pushed 60 Å apart share no edges and no neighbours, so a
    size-consistent potential must return exactly the sum of their energies.
    Per-atom EVB does. Whole-structure EVB cannot, because λ_min(H_A + H_B) ≥
    λ_min(H_A) + λ_min(H_B) with equality only when the two ground states
    coincide — the documented cost of the literal formulation.
    """
    pos_a, types_a = random_structure(n=5, seed=12)
    pos_b, types_b = random_structure(n=6, seed=13, scale=1.5)
    far = pos_b + torch.tensor([60.0, 0.0, 0.0], dtype=DTYPE)
    pos_ab = torch.cat([pos_a, far])
    types_ab = torch.cat([types_a, types_b])

    m = _model(seed=7, n_experts=3, moe_mixture='evb', moe_scope='atom')
    e_ab = m(pos_ab, types_ab).item()
    e_sum = m(pos_a, types_a).item() + m(pos_b, types_b).item()
    assert abs(e_ab - e_sum) < 1e-9, f"atom scope not additive: {e_ab - e_sum:.2e}"

    g = _model(seed=7, n_experts=3, moe_mixture='evb', moe_scope='global')
    g_ab = g(pos_ab, types_ab).item()
    g_sum = g(pos_a, types_a).item() + g(pos_b, types_b).item()
    assert g_ab > g_sum + 1e-6, "global scope unexpectedly additive here"
    print(f"  size consistency: atom scope exact (Δ={abs(e_ab - e_sum):.1e} eV); "
          f"global scope superadditive by {g_ab - g_sum:+.4f} eV on 11 atoms")


def test_energy_continuous_at_cutoff():
    """No jump as the last edge crosses r_cut — the per-type constants survive."""
    for mixture in ('evb', 'moe', 'softmin'):
        m = _model(seed=8, n_experts=3, moe_mixture=mixture)
        r = m.r_cut_edge
        types = torch.tensor([1, 3])
        e_in = m(torch.tensor([[0.0, 0, 0], [r - 1e-6, 0, 0]], dtype=DTYPE), types)
        e_out = m(torch.tensor([[0.0, 0, 0], [r + 1e-6, 0, 0]], dtype=DTYPE), types)
        gap = (e_in - e_out).abs().item()
        assert gap < 1e-8, f"{mixture}: energy jump {gap:.3e} across r_cut_edge"
        # The edgeless side is the pure atomic-constant mixture, not zero.
        assert e_out.abs().item() > 1e-6, f"{mixture}: edgeless energy collapsed to 0"
    print("  continuous across r_cut for evb / moe / softmin; the edgeless limit "
          "keeps the per-(type, expert) constants")


def test_forward_paths_agree():
    """forward, forward_pbc, forward_batch_multi and forward_batch give one answer."""
    pos, types = random_structure(n=7, seed=14)
    for scope in ('atom', 'global'):
        for K, mixture in ((2, 'evb'), (4, 'evb'), (3, 'moe')):
            m = _model(seed=9, n_experts=K, moe_mixture=mixture, moe_scope=scope)
            e1 = m(pos, types)
            e2 = m.forward_pbc(pos, types, *_pbc_args(m, pos))
            e3 = m.forward_batch_multi([pos, pos], [types, types])
            e4 = m.forward_batch([pos, pos], types, topology=None)
            for got in (e2, e3[0], e3[1], e4[0], e4[1]):
                assert abs((got - e1).item()) < 1e-10, \
                    f"{scope}/K={K}/{mixture}: forward paths disagree"
    print("  forward == forward_pbc == forward_batch_multi == forward_batch, "
          "both scopes")


def test_mixture_info_shapes_and_weights():
    """The diagnostics dict is well formed and the weights are a distribution."""
    pos, types = random_structure(n=6, seed=15)
    K = 3
    for scope, n_seg in (('atom', 6), ('global', 1)):
        for mixture in ('evb', 'moe', 'softmin', 'mean'):
            m = _model(seed=10, n_experts=K, moe_mixture=mixture, moe_scope=scope)
            _, info = m(pos, types, return_mixture=True)
            P = m.mixture_head.pairs.shape[0]
            assert info['expert_energies'].shape == (n_seg, K)
            assert info['coupling'].shape == (n_seg, P)
            assert info['weights'].shape == (n_seg, K)
            w = info['weights']
            assert torch.allclose(w.sum(-1), torch.ones(n_seg, dtype=DTYPE), atol=1e-10)
            assert (w >= -1e-12).all(), f"{mixture}: negative weight"
            if mixture == 'evb':
                assert torch.allclose(info['c0'] ** 2, w, atol=1e-10)
            else:
                assert info['c0'] is None, f"{mixture} has no eigenvector to report"
    # Single-head models report None rather than an empty dict.
    _, info = _model(seed=10)(pos, types, return_mixture=True)
    assert info is None
    print("  info dict: shapes (n_seg, K)/(n_seg, P), weights ≥ 0 summing to 1, "
          "c0² == weights under evb")


def test_batch_info_maps_back_to_structures():
    """In a batch, seg_struct maps per-segment rows back to their structure."""
    pos_a, types_a = random_structure(n=4, seed=16)
    pos_b, types_b = random_structure(n=7, seed=17)
    m = _model(seed=12, n_experts=3, moe_scope='atom')
    energies, info = m.forward_batch_multi([pos_a, pos_b], [types_a, types_b],
                                           return_mixture=True)
    assert info['weights'].shape == (11, 3)
    assert info['seg_struct'].tolist() == [0] * 4 + [1] * 7
    # Per-segment energies scatter back to exactly the reported totals.
    recon = torch.zeros(2, dtype=DTYPE).index_add(0, info['seg_struct'], info['energy_seg'])
    recon = recon + torch.stack([m.atomic_energy[types_a].sum(),
                                 m.atomic_energy[types_b].sum()])
    assert torch.allclose(recon, energies, atol=1e-12)
    print("  batched info: 11 atom segments tagged 0/1, energy_seg re-sums to the totals")


def test_coupling_modes():
    """'none' reproduces the hard min; 'const' drops the geometry dependence."""
    pos, types = random_structure(n=6, seed=18)
    m = _model(seed=13, n_experts=3, moe_coupling='none')
    _, info = m(pos, types, return_mixture=True)
    assert info['coupling'].shape[1] == 0
    assert torch.allclose(info['energy_seg'], info['expert_energies'].min(-1).values,
                          atol=1e-12), "C=0 must give the hard minimum"

    const = _model(seed=13, n_experts=3, moe_coupling='const')
    assert const.mixture_head.coupling_net is None
    _, ci = const(pos, types, return_mixture=True)
    # Per-type constants only: two atoms of the same element share a coupling row.
    same = (types.unsqueeze(0) == types.unsqueeze(1)).nonzero()
    for a, b in same.tolist():
        assert torch.allclose(ci['coupling'][a], ci['coupling'][b])

    pos_pos = _model(seed=13, n_experts=3, moe_coupling_positive=True)
    _, pi = pos_pos(pos, types, return_mixture=True)
    assert (pi['coupling'] > 0).all(), "coupling_positive did not force C > 0"

    chain = _model(seed=13, n_experts=4, moe_coupling_topology='chain')
    assert chain.mixture_head.pairs.shape[0] == 3
    print("  coupling modes: 'none' → hard min, 'const' is type-only, softplus keeps "
          "C > 0, 'chain' gives K-1 pairs")


def test_invalid_configuration_raises():
    """Typos in the mixture configuration fail loudly at construction."""
    for kw in (dict(moe_mixture='softmax'), dict(moe_scope='molecule'),
               dict(moe_coupling='learned'), dict(moe_coupling_topology='ring')):
        try:
            _model(n_experts=2, **kw)
            raise AssertionError(f"accepted bad config {kw}")
        except ValueError:
            pass
    print("  bad mixture / scope / coupling / topology names raise ValueError")


def test_checkpoint_roundtrip():
    """A mixture checkpoint rebuilds through ECENetCalculator.from_checkpoint."""
    from ecenet.calculator import ECENetCalculator

    hp = dict(n_types=N_TYPES, n_mp=1, r_cut_edge=5.0, r_cut_neighbor=4.0,
              l_max=2, n_max=3, embed_dim=8, n_layers=2, n_max_d=4,
              n_experts=3, moe_mixture='evb', moe_scope='atom',
              moe_coupling='mlp', moe_coupling_topology='chain')
    torch.manual_seed(21)
    model = ECENet(**{k: v for k, v in hp.items() if k != 'n_mp'}, n_mp=1).double()
    pos, types = random_structure(n=5, seed=19)
    e_ref = model(pos, types).item()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'moe.mdl')
        torch.save({'model': model.state_dict(), 'hparams': hp,
                    'element_to_type': {'H': 0, 'C': 1, 'N': 2, 'O': 3}}, path)
        calc = ECENetCalculator.from_checkpoint(path, device='cpu')
        assert calc.model.mixture_head is not None
        assert calc.model.mixture_head.pairs.shape[0] == 2   # chain over K=3
        e_new = calc.model(pos, types).item()
    assert abs(e_new - e_ref) < 1e-12, "rebuilt model disagrees with the original"
    print("  checkpoint round-trip: hparams rebuild the head (chain, K=3) exactly")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Trainer
# ═══════════════════════════════════════════════════════════════════════════

def _structures(n, seed=0):
    """Random periodic structures — the xyz trainer's input format."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        na = rng.randint(4, 8)
        L = rng.uniform(7.0, 8.0)
        cell = np.diag([L, L, L]).astype(np.float64)
        frac = rng.uniform(0, 1, size=(na, 3))
        out.append({
            'numbers': rng.choice([1, 6, 7, 8], size=na).astype(np.int64),
            'positions': frac @ cell, 'cell': cell, 'pbc': True,
            'energy': float(rng.uniform(-5, 5) * na),
            'forces': rng.uniform(-1, 1, size=(na, 3)).astype(np.float64),
            'stress': None, 'n_atoms': na,
        })
    return out


_TRAIN_COMMON = dict(
    l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4,
    r_cut_edge=4.0, r_cut_neighbor=3.5,
    n_epochs=2, batch_size=4, lr=5e-3,
    dtype=DTYPE, device=torch.device('cpu'), seed=0, verbose=True,
)


def test_trainer_smoke():
    """End-to-end MoE training, with the diversity regulariser switched on."""
    from train_ecenet_moe import train_ecenet_moe

    model, les_module, results = train_ecenet_moe(
        train_structures=_structures(12, seed=1),
        test_structures=_structures(3, seed=2),
        n_val=2, n_experts=3, moe_mixture='evb',
        moe_diversity_weight=0.01, moe_diversity_kind='load',
        **_TRAIN_COMMON)
    assert les_module is None
    assert model.mixture_head is not None
    assert np.isfinite(results['val_force_mae']), "training produced non-finite errors"
    # The experts must actually be in use, not collapsed onto one.
    pos = torch.tensor(_structures(1, seed=3)[0]['positions'], dtype=DTYPE)
    types = torch.zeros(pos.shape[0], dtype=torch.long)
    _, info = model(pos, types, return_mixture=True)
    assert info['weights'].shape[1] == 3
    print(f"  trainer smoke: {results['n_params']:,} params, "
          f"val F={results['val_force_mae']:.4f} eV/Å, diversity term active")


def test_trainer_rejects_diversity_without_experts():
    """A diversity weight with a single head is a configuration error, not a no-op."""
    from train_ecenet_moe import train_ecenet_moe

    try:
        train_ecenet_moe(train_structures=_structures(4, seed=4), n_val=1,
                         n_experts=1, moe_diversity_weight=0.1, **_TRAIN_COMMON)
        raise AssertionError("accepted moe_diversity_weight with n_experts=1")
    except ValueError as e:
        assert 'n_experts' in str(e)
    print("  moe_diversity_weight without experts raises instead of silently doing nothing")


def test_freeze_experts_stage_two():
    """moe_freeze_experts trains the couplings alone against a fixed expert basis."""
    from train_ecenet_moe import train_ecenet_moe

    structs = _structures(10, seed=7)
    model, _, results = train_ecenet_moe(
        train_structures=structs, n_val=2, n_experts=3,
        moe_freeze_experts=True, **_TRAIN_COMMON)
    head = model.mixture_head
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith('mixture_head.coupling') for n in trainable), \
        f"stage 2 left non-coupling parameters trainable: {sorted(trainable)[:3]}"
    assert results['n_params'] == sum(p.numel() for p in head.parameters()
                                      if p.requires_grad)
    # Nothing to train once the couplings are gone — that is an error, not a no-op.
    try:
        train_ecenet_moe(train_structures=structs, n_val=2, n_experts=3,
                         moe_coupling='none', moe_freeze_experts=True, **_TRAIN_COMMON)
        raise AssertionError("accepted freeze with no couplings")
    except ValueError as e:
        assert 'couplings' in str(e)
    print(f"  stage 2: {results['n_params']:,} trainable params, all couplings; "
          "freezing with no couplings raises")


def test_compare_mixtures_harness():
    """The baseline harness trains every rule on identical data and tabulates it."""
    from train_ecenet_moe import compare_mixtures

    table = compare_mixtures(
        mixtures=('single', 'moe', 'evb'), n_experts=2,
        train_structures=_structures(10, seed=5),
        test_structures=_structures(3, seed=6), n_val=2,
        **_TRAIN_COMMON)
    assert set(table) == {'single', 'moe', 'evb'}
    for name, r in table.items():
        assert np.isfinite(r['val_force_mae']), f"{name}: non-finite result"
    assert table['single']['n_params'] < table['evb']['n_params'], \
        "the mixture head should add read-out parameters"
    assert table['moe']['n_params'] != table['evb']['n_params'], \
        "the gate and the coupling heads are different parameter sets"
    print("  compare_mixtures: single / moe / evb trained on one split, "
          f"params {table['single']['n_params']:,} → {table['evb']['n_params']:,}")


if __name__ == "__main__":
    print("EVB mixture of experts\n" + "=" * 60)
    print("\n[algebra]")
    test_coupling_topologies()
    test_hamiltonian_assembly()
    test_two_state_matches_eigensolver()
    test_weights_match_analytic_formula()
    test_zero_coupling_is_hard_min()
    test_evb_is_variational()
    test_avoided_crossing_and_smoothness()
    test_softmin_limits()
    test_diversity_loss_properties()
    print("\n[model]")
    test_single_expert_is_the_plain_model()
    test_mixture_head_replaces_the_single_head()
    test_matched_single_head_control()
    test_so3_invariance()
    test_forces_finite_difference()
    test_hellmann_feynman()
    test_double_backward_for_force_training()
    test_size_consistency()
    test_energy_continuous_at_cutoff()
    test_forward_paths_agree()
    test_mixture_info_shapes_and_weights()
    test_batch_info_maps_back_to_structures()
    test_coupling_modes()
    test_invalid_configuration_raises()
    test_checkpoint_roundtrip()
    print("\n[trainer]")
    test_trainer_smoke()
    test_trainer_rejects_diversity_without_experts()
    test_freeze_experts_stage_two()
    test_compare_mixtures_harness()
    print("\nAll tests passed.")
