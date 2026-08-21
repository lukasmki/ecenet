"""Integration test: MultiECENet constructs, runs, is SO(3)-invariant, and its
EVB mixing is self-consistent (variational bound, eigenvector identity, stable
at a degenerate Hamiltonian) across the shared- and independent-trunk layouts.

Run:  python tests/test_evbmodel.py
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch

from ecenet import MultiECENet

torch.manual_seed(0)
DTYPE = torch.float64
N_TYPES = 4

COMMON = dict(
    n_types=N_TYPES, r_cut_edge=5.0, r_cut_neighbor=4.0,
    l_max=2, n_max=3, embed_dim=8, n_layers=2, n_max_d=4,
)


def random_structure(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g, dtype=DTYPE) * 1.8
    types = torch.randint(0, N_TYPES, (n,), generator=g)
    return pos, types


def rand_rotation(seed=1):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g, dtype=DTYPE)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diag(R))  # fix QR sign ambiguity -> Haar-uniform on O(3).
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _energy_and_forces(model, pos, types):
    p = pos.clone().requires_grad_(True)
    e = model(p, types)
    f = -torch.autograd.grad(e, p, create_graph=True)[0]
    return e, f


def test_constructs_and_runs():
    pos, types = random_structure()
    model = MultiECENet(**COMMON).double()
    e, f = _energy_and_forces(model, pos, types)
    assert e.dim() == 0 and torch.isfinite(e), "energy not a finite scalar"
    assert f.shape == pos.shape and torch.isfinite(f).all(), "bad forces"
    print(f"  MultiECENet runs: E={e.item():.4f}, |F|max={f.abs().max():.3f}")


def test_so3_invariance():
    pos, types = random_structure(seed=2)
    model = MultiECENet(**COMMON).double()
    Q = rand_rotation()
    err = (model(pos, types) - model(pos @ Q.T, types)).abs().item()
    assert err < 1e-9, f"energy not SO(3)-invariant: {err:.2e}"
    print(f"  SO(3) invariance: |E(Rx) - E(x)| = {err:.1e}")


def test_forces_finite_difference():
    pos, types = random_structure(seed=3)
    model = MultiECENet(**COMMON).double()
    _, f = _energy_and_forces(model, pos, types)
    eps = 1e-5
    fd = torch.zeros_like(pos)
    for i in range(pos.shape[0]):
        for d in range(3):
            p = pos.clone(); p[i, d] += eps
            ep = model(p, types)
            p = pos.clone(); p[i, d] -= eps
            em = model(p, types)
            fd[i, d] = -(ep - em) / (2 * eps)
    err = (f - fd).abs().max().item()
    assert err < 1e-5, f"analytic vs FD forces mismatch {err:.2e}"
    print(f"  forces match finite-difference (max err {err:.1e})")


def test_variational_bound():
    """The mixed ground state never sits above the lowest diabat: coupling can
    only push the lowest eigenvalue down."""
    model = MultiECENet(states=[(0, 1), (1, 2), (-1, 2)], **COMMON).double()
    worst = float('-inf')   # least-negative gap seen: the tightest test of the bound
    for seed in range(5):
        pos, types = random_structure(n=6, seed=20 + seed)
        e, H = model(pos, types, return_matrix=True)
        gap = e.item() - H.diagonal().min().item()
        assert gap <= 1e-12, f"E={e.item()} above lowest diabat by {gap:.2e}"
        worst = max(worst, gap)
    print(f"  variational bound holds over 5 geometries (max E - min_k H_kk = {worst:.1e})")


def test_evb_eigenvector_identity():
    """eigvalsh(H)[0] is the standard EVB combination of the same matrix
    elements, E = Σ_k c_k² H_kk + 2 Σ_{k<l} c_k c_l H_kl, and c² is a
    normalized set of diabatic populations."""
    model = MultiECENet(states=[(0, 1), (1, 2), (-1, 2)], **COMMON).double()
    pos, types = random_structure(seed=4)
    e, H = model(pos, types, return_matrix=True)
    c = model.ground_vector(H)
    combined = torch.einsum('k,kl,l->', c, H, c)
    err = (e - combined).abs().item()
    assert err < 1e-10, f"eigenvalue != c^T H c: {err:.2e}"
    w = model.state_weights(pos, types)
    assert abs(w.sum().item() - 1.0) < 1e-12 and (w >= 0).all()
    print(f"  EVB identity |eigvalsh - cᵀHc| = {err:.1e}, weights {w.numpy().round(3)}")


def test_mix_mode_eigvector_matches():
    """mix_mode='eigvector' is the same energy AND the same forces (exact by
    Hellmann-Feynman) as the default eigvalsh path."""
    pos, types = random_structure(seed=5)
    a = MultiECENet(**COMMON).double()
    b = MultiECENet(mix_mode='eigvector', **COMMON).double()
    b.load_state_dict(a.state_dict())
    _, fa = _energy_and_forces(a, pos, types)
    _, fb = _energy_and_forces(b, pos, types)
    de = (a(pos, types) - b(pos, types)).abs().item()
    df = (fa - fb).abs().max().item()
    assert de < 1e-12 and df < 1e-10, f"mix_mode mismatch: dE={de:.2e} dF={df:.2e}"
    print(f"  mix_mode='eigvector' matches eigvalsh (dE={de:.1e}, dF={df:.1e})")


def test_degenerate_hamiltonian_is_stable():
    """H a multiple of the identity (identical diabats, zero coupling) is the
    worst case for eigen-decomposition gradients: the ground eigenvector is
    arbitrary within the degenerate subspace. Energy and forces must still come
    out finite — this is why the off-diagonal heads are not zero-initialised."""
    pos, types = random_structure(seed=6)
    model = MultiECENet(**COMMON).double()
    with torch.no_grad():
        model.heads['1_1'].load_state_dict(model.heads['0_0'].state_dict())
        last = model.heads['0_1'].linears[-1]
        last.weight.zero_(); last.bias.zero_()
    e, H = model(pos, types, return_matrix=True)
    off = H[0, 1].abs().item()
    spread = (H[0, 0] - H[1, 1]).abs().item()
    assert off < 1e-14 and spread < 1e-14, "test did not actually build a degenerate H"
    _, f = _energy_and_forces(model, pos, types)
    assert torch.isfinite(e) and torch.isfinite(f).all(), "degenerate H gave non-finite grads"
    print(f"  degenerate H (off={off:.0e}, split={spread:.0e}): E and forces finite")


def test_independent_trunks():
    """shared_trunk=False gives each diabat its own ECENet; off-diagonal heads
    read both trunks' invariants concatenated."""
    pos, types = random_structure(seed=7)
    model = MultiECENet(shared_trunk=False, **COMMON).double()
    assert len(model.trunks) == 2
    e, f = _energy_and_forces(model, pos, types)
    assert torch.isfinite(e) and torch.isfinite(f).all()
    Q = rand_rotation(seed=4)
    err = (model(pos, types) - model(pos @ Q.T, types)).abs().item()
    assert err < 1e-9, f"independent trunks not SO(3)-invariant: {err:.2e}"
    print(f"  shared_trunk=False runs: E={e.item():.4f}, SO(3) err {err:.1e}")


def test_training_path_forward_batch_multi():
    for shared in (True, False):
        model = MultiECENet(shared_trunk=shared, **COMMON).double()
        structs = [random_structure(n=5 + b, seed=10 + b) for b in range(3)]
        pos_list = [s[0].clone().requires_grad_(True) for s in structs]
        typ_list = [s[1] for s in structs]
        energies = model.forward_batch_multi(pos_list, typ_list)
        assert energies.shape == (3,) and torch.isfinite(energies).all()
        grads = torch.autograd.grad(energies.sum(), pos_list, create_graph=True)
        assert all(torch.isfinite(g).all() for g in grads)
        # Batched energies must equal the per-structure forwards.
        single = torch.stack([model(p, t) for p, t in zip(pos_list, typ_list)])
        err = (energies - single).abs().max().item()
        assert err < 1e-10, f"batch != single (shared_trunk={shared}): {err:.2e}"
        print(f"  forward_batch_multi (shared_trunk={shared}): "
              f"{energies.detach().numpy().round(3)}, vs single {err:.1e}")


def test_forward_batch_fixed_topology():
    """The vectorized fixed-topology path agrees with the per-structure one."""
    model = MultiECENet(**COMMON).double()
    pos, types = random_structure(seed=8)
    ei, ej, nb_src, nb_dst = model.trunks[0]._local_topology(pos)
    topo = dict(edge_i=ei, edge_j=ej, nb_src=nb_src, nb_dst=nb_dst)
    batch = [pos, pos + 0.05]
    fixed = model.forward_batch(batch, types, topology=topo)
    single = torch.stack([model(p, types) for p in batch])
    err = (fixed - single).abs().max().item()
    assert err < 1e-10, f"forward_batch != single: {err:.2e}"
    print(f"  forward_batch (fixed topology) matches per-structure ({err:.1e})")


def test_forward_pbc_matches_open_with_zero_shifts():
    """forward_pbc with zero shift vectors is the non-periodic calculation."""
    model = MultiECENet(**COMMON).double()
    pos, types = random_structure(seed=9)
    ei, ej, nb_src, nb_dst = model.trunks[0]._local_topology(pos)
    z_e = torch.zeros(len(ei), 3, dtype=DTYPE)
    z_n = torch.zeros(len(nb_src), 3, dtype=DTYPE)
    p = pos.clone().requires_grad_(True)
    e_pbc = model.forward_pbc(p, types, ei, ej, z_e, nb_src, nb_dst, z_n)
    f_pbc = -torch.autograd.grad(e_pbc, p)[0]
    err = (e_pbc - model(pos, types)).abs().item()
    assert err < 1e-12 and torch.isfinite(f_pbc).all(), f"pbc != open: {err:.2e}"
    print(f"  forward_pbc(zero shifts) == forward ({err:.1e}), forces finite")


def test_zero_edge_structure():
    """Atoms beyond r_cut_edge: no edges, so H is the per-state atomic baseline
    alone and the energy is its lowest entry."""
    model = MultiECENet(**COMMON).double()
    with torch.no_grad():   # distinct baselines so the ground state is unambiguous
        model.atomic_energy.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0],
                                                [0.5, 1.0, 1.5, 2.0]], dtype=DTYPE))
    pos = torch.tensor([[0., 0., 0.], [100., 0., 0.]], dtype=DTYPE)
    types = torch.tensor([0, 1])
    e, H = model(pos, types, return_matrix=True)
    assert H[0, 1].abs().item() == 0.0, "coupling should vanish with no edges"
    expected = min(1.0 + 2.0, 0.5 + 1.0)
    assert abs(e.item() - expected) < 1e-12, f"{e.item()} != {expected}"
    print(f"  zero-edge structure: E = min per-state baseline = {e.item():.3f}")


SECTORS = [(-1, 1), (-1, 1), (0, 1), (0, 1), (1, 1), (1, 1)]


def test_charge_sector_selects_the_right_block():
    """A structure declaring charge Q mixes only the diabats labelled Q, and the
    result is identical to a standalone model built from that block alone."""
    pos, types = random_structure(seed=11)
    m = MultiECENet(states=SECTORS, **COMMON).double()
    assert m.sectors == {(-1, 1): [0, 1], (0, 1): [2, 3], (1, 1): [4, 5]}
    with torch.no_grad():   # separate the sectors so a mix-up would be obvious
        m.atomic_energy.copy_(torch.tensor(
            [[-1.] * N_TYPES, [-1.] * N_TYPES, [0.] * N_TYPES,
             [0.] * N_TYPES, [2.] * N_TYPES, [2.] * N_TYPES], dtype=DTYPE))

    sub = MultiECENet(states=[(0, 1), (0, 1)], **COMMON).double()
    sub.trunks.load_state_dict(m.trunks.state_dict())
    for dst, src in (('0_0', '2_2'), ('1_1', '3_3'), ('0_1', '2_3')):
        sub.heads[dst].load_state_dict(m.heads[src].state_dict())
    with torch.no_grad():
        sub.atomic_energy.copy_(m.atomic_energy[2:4])
    err = (m(pos, types, charge=0) - sub(pos, types)).abs().item()
    assert err < 1e-12, f"sector block != standalone model: {err:.2e}"

    w = m.state_weights(pos, types, charge=0)
    assert w[[0, 1, 4, 5]].abs().max().item() < 1e-12, "out-of-sector states carry weight"
    assert abs(w.sum().item() - 1.0) < 1e-12
    print(f"  charge sector q=0 == standalone 2-state model ({err:.1e}), "
          f"weights {w.numpy().round(3)}")


def test_out_of_sector_parameters_get_no_gradient():
    """Masked diabats are decoupled AND lifted, so they cannot reach the ground
    state and their heads receive exactly zero gradient."""
    pos, types = random_structure(seed=12)
    m = MultiECENet(states=SECTORS, **COMMON).double()
    e = m(pos, types, charge=0)
    out_of_sector = [m.heads['0_0'].linears[-1].weight,
                     m.heads['4_5'].linears[-1].weight]
    in_sector = [m.heads['2_2'].linears[-1].weight,
                 m.heads['2_3'].linears[-1].weight]
    g_out = torch.autograd.grad(e, out_of_sector, retain_graph=True, allow_unused=True)
    g_in = torch.autograd.grad(e, in_sector, allow_unused=True)
    assert all(g is None or g.abs().max() == 0 for g in g_out), "out-of-sector head got gradient"
    assert all(g is not None and g.abs().max() > 0 for g in g_in), "in-sector head got none"
    print("  out-of-sector heads receive exactly zero gradient; in-sector heads do not")


def test_mixed_sector_batch():
    """One batch may hold structures from different sectors: the matrix stays a
    uniform (B, S, S) and a single batched eigvalsh serves them all."""
    m = MultiECENet(states=SECTORS, **COMMON).double()
    pos, types = random_structure(seed=13)
    charges = [-1, 0, 1]
    pos_list = [pos.clone().requires_grad_(True) for _ in charges]
    typ_list = [types] * len(charges)
    batched = m.forward_batch_multi(pos_list, typ_list, charge=charges)
    single = torch.stack([m(p, types, charge=q) for p, q in zip(pos_list, charges)])
    err = (batched - single).abs().max().item()
    assert err < 1e-12, f"mixed-sector batch != per-structure: {err:.2e}"
    grads = torch.autograd.grad(batched.sum(), pos_list)
    assert all(torch.isfinite(g).all() for g in grads)
    print(f"  mixed-sector batch matches per-structure forwards ({err:.1e})")


def test_spin_sector_and_unknown_sector_error():
    """Multiplicity filters on the same footing as charge, and an unrepresented
    sector fails loudly instead of silently mixing the wrong states."""
    pos, types = random_structure(seed=14)
    m = MultiECENet(states=[(0, 1), (0, 1), (0, 3), (0, 3)], **COMMON).double()
    singlet = m(pos, types, charge=0, spin=1)
    triplet = m(pos, types, charge=0, spin=3)
    assert (singlet - triplet).abs().item() > 1e-9, "spin did not select a different block"
    # charge alone must not filter on multiplicity: all four states stay active
    w_all = m.state_weights(pos, types, charge=0)
    assert (w_all > 0).sum().item() > 2, "charge-only filter wrongly narrowed by spin"
    try:
        m(pos, types, charge=+7)
    except ValueError as exc:
        assert 'sector' in str(exc)
    else:
        raise AssertionError("unknown sector should raise")
    print("  spin selects its own block; unknown sector raises")


def test_no_charge_is_backward_compatible():
    """Passing no charge leaves every diabat active — the pre-sector behaviour."""
    pos, types = random_structure(seed=15)
    m = MultiECENet(states=SECTORS, **COMMON).double()
    e_all, H = m(pos, types, return_matrix=True)
    assert H.shape == (6, 6) and torch.isfinite(H).all()
    assert (H.diagonal().abs() < 1e5).all(), "unmasked run should not lift any diagonal"
    assert abs(e_all.item() - torch.linalg.eigvalsh(H)[0].item()) < 1e-12
    print("  charge=None keeps all 6 diabats active (no masking)")


def test_force_loss_double_backward_through_masked_sectors():
    """The path training actually takes: force loss -> backward, i.e. a SECOND
    derivative through eigvalsh. Masked diabats must not be mutually degenerate,
    or eigvalsh's double backward hits 1/(lambda_i - lambda_j) = 1/0 and every
    parameter gradient becomes NaN."""
    m = MultiECENet(states=SECTORS, **COMMON).double()
    pos, types = random_structure(seed=16)
    charges = [-1, 0, 1]
    pos_list = [pos.clone().requires_grad_(True) for _ in charges]
    energies = m.forward_batch_multi(pos_list, [types] * 3, charge=charges)
    grads = torch.autograd.grad(energies.sum(), pos_list, create_graph=True)
    force_loss = sum((-g - 0.1).pow(2).mean() for g in grads)
    force_loss.backward()
    pgrads = [p.grad for p in m.parameters() if p.grad is not None]
    assert pgrads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in pgrads), \
        "NaN/Inf parameter gradient from the force-loss double backward"
    lifted = m._MASKED_DIAG * torch.arange(1, m.n_states + 1)
    assert len(set(lifted.tolist())) == m.n_states, "masked diagonals must be distinct"
    print(f"  force-loss double backward through masked sectors is finite "
          f"({len(pgrads)} param grads, max {max(g.abs().max().item() for g in pgrads):.2e})")


if __name__ == "__main__":
    print("MultiECENet (EVB) integration")
    test_constructs_and_runs()
    test_so3_invariance()
    test_forces_finite_difference()
    test_variational_bound()
    test_evb_eigenvector_identity()
    test_mix_mode_eigvector_matches()
    test_degenerate_hamiltonian_is_stable()
    test_independent_trunks()
    test_training_path_forward_batch_multi()
    test_forward_batch_fixed_topology()
    test_forward_pbc_matches_open_with_zero_shifts()
    test_zero_edge_structure()
    test_charge_sector_selects_the_right_block()
    test_out_of_sector_parameters_get_no_gradient()
    test_mixed_sector_batch()
    test_spin_sector_and_unknown_sector_error()
    test_no_charge_is_backward_compatible()
    test_force_loss_double_backward_through_masked_sectors()
    print("All tests passed.")
