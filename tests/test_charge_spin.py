"""Tests for total-charge / total-spin conditioning (ecenet/electronic.py),
wired into ECENet via ``charge_spin=True`` and passed per call as
``forward(..., total_charge=, total_spin=)``.

Three layers, in order:

  1. the state vector on plain tensors — the extensive/intensive layout, the
     accepted input forms, and the size-consistency property that motivates
     `Q/N` being in there at all;
  2. the model integration — ``charge_spin=False`` is bit-for-bit the old
     model (same parameters, same state-dict keys, same energy), the enabled
     model is identity at init, a trained one actually separates charge and
     spin states, the energy stays SO(3)-invariant and continuous at r_cut with
     the conditioning active, forces match finite differences and double
     backward works (so force training is possible), all four forward paths
     agree, and an *edgeless* ion is still distinguishable from a neutral atom;
  3. the ASE calculator — reading the state off `atoms.info` / the per-atom
     arrays / a fixed calculator override, and the cache invalidation that
     makes ``atoms.info['charge'] = 1`` actually take effect.

The load-bearing claims are (a) the default model is untouched, (b) the gate
is invariant so equivariance survives, and (c) the calculator does not serve a
neutral energy for a charged request out of ASE's result cache.

Run:  python tests/test_charge_spin.py     (from the repo root)
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

import numpy as np
import torch

from ecenet import ECENet
from ecenet.electronic import N_STATE_FEATURES, StateAtomicEnergy, state_features
from ecenet.radial import find_edges

torch.manual_seed(0)
DTYPE = torch.float64
N_TYPES = 4
COMMON = dict(
    n_types=N_TYPES, r_cut_edge=5.0, r_cut_neighbor=4.0,
    l_max=2, n_max=3, embed_dim=8, n_layers=2, n_max_d=4,
)


# ── helpers ──────────────────────────────────────────────────────────────

def random_structure(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g, dtype=DTYPE) * 1.8
    types = torch.randint(0, N_TYPES, (n,), generator=g)
    return pos, types


def rand_rotation(seed=1):
    g = torch.Generator().manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=DTYPE))
    Q = Q * torch.sign(torch.diag(R))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def model(seed=0, **kw):
    """A charge/spin model with its state heads *activated*.

    Both heads are zero-init by design, so a fresh model ignores the state
    entirely — which is the point of `test_identity_at_init`, and useless for
    every test that wants to see a response. Perturbing the last layers is the
    cheapest stand-in for training.
    """
    torch.manual_seed(seed)
    m = ECENet(**COMMON, charge_spin=True, **kw).to(DTYPE)
    activate(m, seed)
    return m


def activate(m, seed=0):
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for head in (m.state_film, m.state_energy):
            if head is None:
                continue
            last = head.mlp[-1]
            last.weight.copy_(torch.randn(last.weight.shape, generator=g,
                                          dtype=DTYPE) * 0.05)
            last.bias.copy_(torch.randn(last.bias.shape, generator=g,
                                        dtype=DTYPE) * 0.05)
    return m


def topology(m, pos):
    """(edge_i, edge_j, nb_src, nb_dst) for one structure, as the model builds them."""
    ei, ej = find_edges(pos, m.r_cut_edge)
    edge_i, edge_j = torch.cat([ei, ej]), torch.cat([ej, ei])
    d = torch.cdist(pos, pos)
    nb_src, nb_dst = ((d < m.r_cut_neighbor) & (d > 1e-10)).nonzero(as_tuple=True)
    return edge_i, edge_j, nb_src, nb_dst


# ═══ 1. the state vector ═════════════════════════════════════════════════

def test_state_vector_layout():
    """[Q, S, Q/N, S/N], in that order, for the documented input forms."""
    sf = state_features(2.0, 1.0, (4,), torch.device('cpu'), DTYPE)
    assert sf.shape == (1, N_STATE_FEATURES)
    assert torch.allclose(sf[0], torch.tensor([2.0, 1.0, 0.5, 0.25], dtype=DTYPE))

    # per-structure vectors, and a scalar broadcast over the batch
    sf = state_features([1.0, -1.0], [0.0, 2.0], (2, 4), torch.device('cpu'), DTYPE)
    assert torch.allclose(sf[:, 2], torch.tensor([0.5, -0.25], dtype=DTYPE))
    sf = state_features(1.0, 0.0, (2, 4), torch.device('cpu'), DTYPE)
    assert torch.allclose(sf[:, 0], torch.ones(2, dtype=DTYPE))
    assert torch.allclose(sf[:, 2], torch.tensor([0.5, 0.25], dtype=DTYPE))

    try:
        state_features([1.0, 2.0, 3.0], None, (2, 4), torch.device('cpu'), DTYPE)
    except ValueError as e:
        assert 'per-structure' in str(e), e
    else:
        raise AssertionError("expected a ValueError on a mismatched batch size")
    print("  state vector [Q, S, Q/N, S/N] and its input forms")


def test_intensive_half_is_size_consistent():
    """Q/N is what survives replicating a system together with its charge.

    Two copies of a +1 ion at 2·N atoms with Q = +2 see the same intensive
    features as one copy — the property that keeps the per-atom conditioning
    from drifting with system size.
    """
    one = state_features(1.0, 1.0, (5,), torch.device('cpu'), DTYPE)
    two = state_features(2.0, 2.0, (10,), torch.device('cpu'), DTYPE)
    assert torch.allclose(one[:, 2:], two[:, 2:])
    assert not torch.allclose(one[:, :2], two[:, :2])   # extensive half does move
    print("  Q/N, S/N invariant under replication; Q, S are not")


def test_state_atomic_energy_zero_at_init():
    head = StateAtomicEnergy(N_TYPES).to(DTYPE)
    types = torch.arange(N_TYPES)
    sf = state_features(1.0, 2.0, (N_TYPES,), torch.device('cpu'), DTYPE)
    out = head(types, sf.expand(N_TYPES, -1))
    assert out.shape == (N_TYPES,)
    assert torch.count_nonzero(out) == 0
    print("  StateAtomicEnergy is exactly zero at init")


# ═══ 2. model integration ════════════════════════════════════════════════

def test_disabled_model_is_untouched():
    """charge_spin=False costs nothing: same parameters, same state-dict keys,
    same energy. A regression here would break every existing checkpoint."""
    torch.manual_seed(1)
    base = ECENet(**COMMON).to(DTYPE)
    torch.manual_seed(1)
    off = ECENet(**COMMON, charge_spin=False).to(DTYPE)
    assert base.state_dict().keys() == off.state_dict().keys()
    assert not any(k.startswith('state_') for k in base.state_dict())
    assert base.state_film is None and base.state_energy is None
    pos, types = random_structure(seed=2)
    assert torch.equal(base(pos, types), off(pos, types))

    torch.manual_seed(1)
    on = ECENet(**COMMON, charge_spin=True).to(DTYPE)
    new = set(on.state_dict()) - set(base.state_dict())
    # 'film_m0' is the m=0 selector buffer, shared with the element gate's shift
    # head and registered here because the state gate emits a shift by default.
    assert new and all(k.startswith('state_') or k == 'film_m0' for k in new), new
    print(f"  charge_spin=False identical; =True adds {len(new)} state keys, "
          f"{sum(p.numel() for p in on.parameters()) - sum(p.numel() for p in base.parameters())} params")


def test_identity_at_init():
    """Both heads are zero-init, so a fresh charge_spin model is state-blind —
    switching the flag on does not move a model's step-0 predictions."""
    pos, types = random_structure(seed=3)
    torch.manual_seed(1)
    base = ECENet(**COMMON).to(DTYPE)
    torch.manual_seed(1)
    on = ECENet(**COMMON, charge_spin=True).to(DTYPE)
    e_ref = base(pos, types)
    for q, s in ((0.0, 0.0), (1.0, 0.0), (-2.0, 3.0)):
        e = on(pos, types, total_charge=q, total_spin=s)
        assert torch.allclose(e, e_ref, atol=1e-12), (q, s, (e - e_ref).item())
    print("  identity at init: the state moves nothing, and matches the plain model")


def test_states_separate_once_trained():
    """An activated model gives genuinely different energies AND forces per state."""
    pos, types = random_structure(seed=4)
    m = model(seed=5)
    p = pos.clone().requires_grad_(True)
    out = {}
    for label, q, s in (('neutral', 0.0, 0.0), ('cation', 1.0, 0.0),
                        ('anion', -1.0, 0.0), ('triplet', 0.0, 2.0)):
        e = m(p, types, total_charge=q, total_spin=s)
        f = -torch.autograd.grad(e, p, retain_graph=True)[0]
        out[label] = (e.item(), f)
    for label in ('cation', 'anion', 'triplet'):
        de = abs(out[label][0] - out['neutral'][0])
        df = (out[label][1] - out['neutral'][1]).abs().max().item()
        assert de > 1e-6 and df > 1e-8, (label, de, df)
        print(f"  {label:8s} ΔE = {de:.4f}  max|ΔF| = {df:.4f}")


def test_each_site_alone_conditions():
    """Either site alone is enough to make the energy state-dependent, and the
    atomic one is the site that survives with no edges."""
    pos, types = random_structure(seed=6)
    for film, atomic in ((True, False), (False, True)):
        m = model(seed=7, charge_spin_film=film, charge_spin_atomic=atomic)
        d = abs(m(pos, types, total_charge=1.0).item() - m(pos, types).item())
        assert d > 1e-8, (film, atomic, d)
        print(f"  film={film}, atomic={atomic}: ΔE(+1) = {d:.4f}")


def test_isolated_ion_still_charge_dependent():
    """Every neighbour beyond r_cut: no edges, so only the atomic term is left.

    Without it a lone Na+ and a lone Na would be exactly degenerate — and the
    energy would jump discontinuously as the last edge crossed the cutoff.
    """
    far = torch.tensor([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=DTYPE)
    types = torch.tensor([0, 1])
    m = model(seed=8)
    e0 = m(far, types).item()
    e1 = m(far, types, total_charge=1.0).item()
    assert abs(e1 - e0) > 1e-8, (e0, e1)
    # and the edgeless path is what ran: no edges within r_cut_edge
    assert len(find_edges(far, m.r_cut_edge)[0]) == 0
    print(f"  edgeless ion: E(0) = {e0:.4f}, E(+1) = {e1:.4f}")


def test_so3_invariance_with_active_state():
    """The state vector is an invariant scalar, so conditioning on it cannot
    break rotational invariance of the energy."""
    pos, types = random_structure(seed=9)
    m = model(seed=10, charge_spin_per_m=True)
    R = rand_rotation()
    for q, s in ((1.0, 0.0), (-1.0, 2.0)):
        e = m(pos, types, total_charge=q, total_spin=s)
        e_rot = m(pos @ R.T, types, total_charge=q, total_spin=s)
        assert torch.allclose(e, e_rot, atol=1e-10), (q, s, (e - e_rot).item())
    print("  SO(3)-invariant with an active per-(channel, m) state gate")


def test_translation_invariance():
    pos, types = random_structure(seed=11)
    m = model(seed=12)
    e = m(pos, types, total_charge=1.0)
    e_t = m(pos + torch.tensor([3.1, -2.0, 0.7], dtype=DTYPE), types, total_charge=1.0)
    assert torch.allclose(e, e_t, atol=1e-10)
    print("  translation-invariant")


def test_forces_finite_difference():
    """Autograd forces still match central differences with the state active —
    the conditioning enters as a smooth multiplicative/additive factor, not as
    a branch on the geometry."""
    pos, types = random_structure(n=5, seed=13)
    m = model(seed=14)
    p = pos.clone().requires_grad_(True)
    e = m(p, types, total_charge=1.0, total_spin=1.0)
    f = -torch.autograd.grad(e, p)[0]
    h = 1e-5
    fd = torch.zeros_like(pos)
    for i in range(pos.shape[0]):
        for a in range(3):
            pp, pm = pos.clone(), pos.clone()
            pp[i, a] += h
            pm[i, a] -= h
            fd[i, a] = -(m(pp, types, total_charge=1.0, total_spin=1.0)
                         - m(pm, types, total_charge=1.0, total_spin=1.0)) / (2 * h)
    err = (f - fd).abs().max().item()
    assert err < 1e-7, err
    print(f"  forces vs central differences: max err {err:.2e}")


def test_double_backward_for_force_training():
    """Force training differentiates the forces again; the state heads must not
    block that."""
    pos, types = random_structure(n=5, seed=15)
    m = model(seed=16)
    p = pos.clone().requires_grad_(True)
    e = m(p, types, total_charge=-1.0)
    f = -torch.autograd.grad(e, p, create_graph=True)[0]
    g = torch.autograd.grad(f.pow(2).sum(), list(m.parameters()), allow_unused=True)
    touched = [x for x in g if x is not None and torch.isfinite(x).all()
               and x.abs().sum() > 0]
    assert touched, "no parameter received a second-order gradient"
    print(f"  double backward reaches {len(touched)} parameter tensors")


def test_energy_continuous_at_cutoff():
    """A neighbour crossing r_cut_edge must not step the energy, state or no state.

    The gate rides on the edge features, which already vanish at the cutoff;
    the atomic term has no edges at all. So the conditioning is continuous for
    the same reason the base read-out is.
    """
    m = model(seed=17)
    types = torch.tensor([0, 1])
    eps = 1e-6
    for q in (0.0, 1.0):
        inside = torch.tensor([[0.0, 0.0, 0.0], [m.r_cut_edge - eps, 0.0, 0.0]],
                              dtype=DTYPE)
        outside = torch.tensor([[0.0, 0.0, 0.0], [m.r_cut_edge + eps, 0.0, 0.0]],
                               dtype=DTYPE)
        jump = abs(m(inside, types, total_charge=q).item()
                   - m(outside, types, total_charge=q).item())
        assert jump < 1e-9, (q, jump)
    print("  energy continuous across r_cut_edge at Q = 0 and Q = +1")


def test_forward_paths_agree():
    """forward / forward_pbc / forward_batch (fixed topology) / forward_batch_multi
    must give the same energy for the same structure and state — including a
    batch where the structures differ *only* in their charge."""
    m = model(seed=18)
    pos, types = random_structure(n=7, seed=19)
    edge_i, edge_j, nb_src, nb_dst = topology(m, pos)
    she = torch.zeros(len(edge_i), 3, dtype=DTYPE)
    shn = torch.zeros(len(nb_src), 3, dtype=DTYPE)
    qs, ss = [0.0, 1.0, -2.0], [0.0, 1.0, 0.0]

    ref = torch.stack([m(pos, types, total_charge=q, total_spin=s)
                       for q, s in zip(qs, ss)])
    pbc = torch.stack([m.forward_pbc(pos, types, edge_i, edge_j, she,
                                     nb_src, nb_dst, shn,
                                     total_charge=q, total_spin=s)
                       for q, s in zip(qs, ss)])
    topo = dict(edge_i=edge_i, edge_j=edge_j, nb_src=nb_src, nb_dst=nb_dst)
    fixed = m.forward_batch([pos] * 3, types, topology=topo,
                            total_charge=qs, total_spin=ss)
    multi = m.forward_batch_multi([pos] * 3, [types] * 3,
                                  total_charge=qs, total_spin=ss)
    free = m.forward_batch([pos] * 3, types, topology=None,
                           total_charge=qs, total_spin=ss)
    for name, got in (('forward_pbc', pbc), ('forward_batch(fixed)', fixed),
                      ('forward_batch_multi', multi), ('forward_batch(free)', free)):
        err = (got - ref).abs().max().item()
        assert err < 1e-10, (name, err)
    # the batch really did carry three different states
    assert (ref - ref[0]).abs().max() > 1e-6
    print("  all four forward paths agree on a mixed-charge batch")


def test_batch_rows_are_per_structure():
    """A (B,) state must be applied row-wise, not broadcast from row 0 — the
    failure mode a same-structure batch would hide."""
    m = model(seed=20)
    s0, t0 = random_structure(n=5, seed=21)
    s1, t1 = random_structure(n=8, seed=22)
    qs = [1.0, -1.0]
    batched = m.forward_batch_multi([s0, s1], [t0, t1], total_charge=qs)
    single = torch.stack([m(s0, t0, total_charge=qs[0]),
                          m(s1, t1, total_charge=qs[1])])
    assert (batched - single).abs().max() < 1e-10
    swapped = m.forward_batch_multi([s0, s1], [t0, t1], total_charge=qs[::-1])
    assert (swapped - batched).abs().max() > 1e-6
    print("  per-structure states land on their own structures")


def test_state_reaches_message_passing_and_mixture():
    """The gate sits in front of the whole stack, so the message-passing layers
    and the mixture read-out see conditioned features too."""
    pos, types = random_structure(n=7, seed=23)
    for extra in (dict(n_mp=2), dict(n_experts=2, moe_mixture='evb')):
        m = model(seed=24, **extra)
        d = abs(m(pos, types, total_charge=1.0).item() - m(pos, types).item())
        assert d > 1e-8, (extra, d)
        print(f"  {list(extra)[0]}: ΔE(+1) = {d:.4f}")


def test_state_blind_model_warns_once():
    """A model that cannot use a charge must say so — silently returning the
    neutral energy under a charged label is the exact failure this feature
    exists to prevent — but not once per step of an MD run."""
    pos, types = random_structure(seed=25)
    m = ECENet(**COMMON).to(DTYPE)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m(pos, types)
        m(pos, types, total_charge=0.0, total_spin=0.0)   # neutral: no warning
        assert not w, [str(x.message) for x in w]
        m(pos, types, total_charge=1.0)
        m(pos, types, total_spin=2.0)
        assert len(w) == 1, [str(x.message) for x in w]
        assert 'charge_spin=False' in str(w[0].message)
    print("  state-blind model warns exactly once on a nonzero state")


def test_invalid_configuration_raises_and_warns():
    try:
        ECENet(**COMMON, charge_spin=True, charge_spin_film=False,
               charge_spin_atomic=False)
    except ValueError as e:
        assert 'charge_spin' in str(e), e
    else:
        raise AssertionError("expected a ValueError with both sites off")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ECENet(**COMMON, charge_spin=False, charge_spin_per_m=True)
        assert any('charge_spin_per_m' in str(x.message) for x in w), \
            [str(x.message) for x in w]
    print("  both-sites-off raises; configured-but-off flags warn")


def test_checkpoint_roundtrip():
    """hparams → ECENet(**hparams) → load_state_dict must reproduce the model,
    charged predictions included."""
    pos, types = random_structure(seed=26)
    m = model(seed=27, charge_spin_per_m=True, charge_spin_embed_dim=8)
    hp = dict(COMMON, charge_spin=True, charge_spin_per_m=True,
              charge_spin_embed_dim=8)
    clone = ECENet(**hp).to(DTYPE)
    clone.load_state_dict(m.state_dict())
    for q in (0.0, 1.0, -1.0):
        assert torch.allclose(m(pos, types, total_charge=q),
                              clone(pos, types, total_charge=q), atol=1e-12)
    print("  checkpoint round-trip reproduces charged predictions")


# ═══ 3. the ASE calculator ═══════════════════════════════════════════════

def _calc_and_atoms():
    from ase import Atoms

    from ecenet.calculator import ECENetCalculator

    m = model(seed=28)
    e2t = {'H': 0, 'C': 1, 'N': 2, 'O': 3}
    calc = ECENetCalculator(m, dtype=DTYPE, element_to_type=e2t)
    atoms = Atoms('OH2', positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0],
                                    [-0.24, 0.93, 0.0]])
    return calc, atoms


def test_calculator_reads_state_from_info():
    calc, atoms = _calc_and_atoms()

    def energy(**info):
        a = atoms.copy()
        a.info.update(info)
        a.calc = calc
        return a.get_potential_energy()

    e0 = energy()
    e_cat = energy(charge=1)
    e_mult = energy(multiplicity=3)
    e_spin = energy(spin=2)
    assert abs(e_cat - e0) > 1e-8, (e0, e_cat)
    assert abs(e_mult - e0) > 1e-8
    # multiplicity 3 ≡ 2 unpaired electrons
    assert abs(e_mult - e_spin) < 1e-12, (e_mult, e_spin)
    # and the neutral energy is still reproducible afterwards
    assert abs(energy() - e0) < 1e-12
    print(f"  info: neutral {e0:.4f}, +1 {e_cat:.4f}, triplet {e_mult:.4f}")


def test_calculator_cache_invalidates_on_state():
    """ASE's check_state ignores atoms.info, so without the override a charged
    request would be served the cached neutral result."""
    calc, atoms = _calc_and_atoms()
    a = atoms.copy()
    a.calc = calc
    e0 = a.get_potential_energy()
    b = atoms.copy()
    b.info['charge'] = 1
    b.calc = calc
    assert 'electronic_state' in calc.check_state(b)
    assert abs(b.get_potential_energy() - e0) > 1e-8
    print("  changing atoms.info['charge'] invalidates the ASE result cache")


def test_calculator_per_atom_arrays_and_override():
    calc, atoms = _calc_and_atoms()
    a = atoms.copy()
    a.set_initial_charges([-1.0, 0.0, 0.0])
    a.calc = calc
    assert calc._read_state(a)['total_charge'] == -1.0
    e_anion = a.get_potential_energy()

    b = atoms.copy()
    b.set_initial_magnetic_moments([1.0, 1.0, 0.0])
    b.calc = calc
    assert calc._read_state(b)['total_spin'] == 2.0

    # a fixed override beats whatever the Atoms object says
    from ecenet.calculator import ECENetCalculator
    forced = ECENetCalculator(calc.model, dtype=DTYPE,
                              element_to_type=calc.element_to_type, charge=-1.0)
    c = atoms.copy()
    c.calc = forced
    assert abs(c.get_potential_energy() - e_anion) < 1e-12
    print("  initial_charges / magmoms read; calculator override wins")


def test_calculator_forces_and_stress_carry_the_state():
    calc, atoms = _calc_and_atoms()
    neutral = atoms.copy()
    neutral.calc = calc
    f0 = neutral.get_forces()
    cation = atoms.copy()
    cation.info['charge'] = 1
    cation.calc = calc
    assert np.abs(cation.get_forces() - f0).max() > 1e-8

    # periodic: stress comes from the strain pass, which must see the state too
    pn, pc = atoms.copy(), atoms.copy()
    for a in (pn, pc):
        a.set_cell([10.0, 10.0, 10.0])
        a.pbc = True
        a.calc = calc
    pc.info['charge'] = 1
    s0, s1 = pn.get_stress(), pc.get_stress()
    assert s0.shape == (6,) and np.abs(s1 - s0).max() > 1e-10
    print(f"  forces and stress respond to the state (max Δσ {np.abs(s1-s0).max():.2e})")


if __name__ == "__main__":
    print("charge/spin conditioning tests\n\n[state vector]")
    test_state_vector_layout()
    test_intensive_half_is_size_consistent()
    test_state_atomic_energy_zero_at_init()
    print("\n[model]")
    test_disabled_model_is_untouched()
    test_identity_at_init()
    test_states_separate_once_trained()
    test_each_site_alone_conditions()
    test_isolated_ion_still_charge_dependent()
    test_so3_invariance_with_active_state()
    test_translation_invariance()
    test_forces_finite_difference()
    test_double_backward_for_force_training()
    test_energy_continuous_at_cutoff()
    test_forward_paths_agree()
    test_batch_rows_are_per_structure()
    test_state_reaches_message_passing_and_mixture()
    test_state_blind_model_warns_once()
    test_invalid_configuration_raises_and_warns()
    test_checkpoint_roundtrip()
    print("\n[calculator]")
    test_calculator_reads_state_from_info()
    test_calculator_cache_invalidates_on_state()
    test_calculator_per_atom_arrays_and_override()
    test_calculator_forces_and_stress_carry_the_state()
    print("\nAll tests passed.")
