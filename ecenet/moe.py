"""ecenet/moe.py — mixture-of-experts read-outs, including the EVB mixture.

ECENet's read-out is a single scalar head over the per-edge invariants. This
module replaces that head with **K expert heads plus a mixing rule**, so one
model carries several *diabatic* energy surfaces and combines them into the
physical one.

The headline rule is **EVB** (empirical valence bond): build a configuration-
dependent real-symmetric Hamiltonian from the expert energies and learned
couplings,

    H_ij(R) = δ_ij V_i(R) + (1-δ_ij) C_ij(R),

and take the lowest eigenvalue as the potential,

    V_EVB(R) = λ_min[H(R)],     w_i(R) = c_0i(R)²,

with the ground-state eigenvector's squared coefficients acting as emergent
expert weights. Because the energy is a plain scalar function of the
positions, autograd forces are exactly -∇V and Hellmann–Feynman holds,

    F = -c_0ᵀ (∇H) c_0 = -Σ_i w_i ∇V_i - Σ_{i≠j} c_0i c_0j ∇C_ij,

so no separate force-gating network is needed and energy conservation is
structural. For K=2 the eigenvalue is closed form,

    V = (V_A+V_B)/2 - sqrt(((V_A-V_B)/2)² + C²),

a hyperbolic regularisation of min(V_A, V_B): far from a crossing the lower
expert dominates, near one the coupling opens an avoided crossing. With C ≡ 0
EVB collapses to the hard minimum, and unlike an ordinary mixture the coupled
energy is *below* every expert (variational), not a convex average of them.

Baselines mixing the same K experts, for controlled comparison:

    'moe'      ordinary MoE — softmax gate over a shared representation,
               V = Σ_i w_i V_i  (soft state selection)
    'softmin'  V = -τ log Σ_i exp(-V_i/τ)  (entropic smoothing of min)
    'mean'     V = (1/K) Σ_i V_i  (no gating at all)
    'evb'      λ_min of the coupled Hamiltonian (coupled variational selection)

All four share the identical expert heads and differ only in `mixture`, so an
ablation changes one string.

Scope
-----
`scope='atom'` (default) builds one K×K Hamiltonian **per atom** and sums the
per-atom ground states; `scope='global'` builds one Hamiltonian for the whole
structure, which is the literal formulation of the theory note. The default
deviates deliberately: λ_min is subadditive, λ_min(H_A + H_B) ≥ λ_min(H_A) +
λ_min(H_B), so the global scope is **not size-consistent** — two non-interacting
subsystems do not give the sum of their energies, and the mixing weights of a
100-atom cell are set by whole-cell energy differences that grow with N. The
per-atom scope is exactly additive, keeps the read-out local like the rest of
the model, and makes the coupling an intensive per-atom quantity. Use
`scope='global'` when reproducing the theory note or for fixed-size systems;
`tests/test_moe.py` asserts both behaviours.

Expert collapse
---------------
Nothing here stops one expert from sitting below all the others everywhere, at
which point c_0 → (1,0,…,0) and the rest are dead weight. `diversity_loss`
supplies the usual counter-pressures (load balancing / usage entropy) for a
trainer to add to the data loss; `scripts/train_ecenet_moe.py` wires it up.
"""

import torch
import torch.nn as nn

from ecenet.radial import get_cutoff_fn, radial_basis

MIXTURES = ('evb', 'moe', 'softmin', 'mean')
COUPLINGS = ('mlp', 'const', 'none')
TOPOLOGIES = ('full', 'chain', 'none')
SCOPES = ('atom', 'global')


# ---------------------------------------------------------------------------
# Coupling topology
# ---------------------------------------------------------------------------

def coupling_pairs(n_experts: int, topology: str = 'full') -> torch.Tensor:
    """Coupled expert pairs (i<j) as a (P, 2) long tensor.

    'full'  every pair — P = K(K-1)/2
    'chain' nearest neighbours only, 0–1, 1–2, … — P = K-1, the sparse expert
            graph of the theory note (§7): a tridiagonal H, cheap for large K
            and enough to interpolate an ordered sequence of regimes
    'none'  no couplings — P = 0, so H is diagonal and EVB degenerates to the
            hard minimum over experts
    """
    if topology not in TOPOLOGIES:
        raise ValueError(f"coupling topology must be one of {TOPOLOGIES}, got {topology!r}")
    if topology == 'none' or n_experts < 2:
        return torch.zeros(0, 2, dtype=torch.long)
    if topology == 'chain':
        i = torch.arange(n_experts - 1)
        return torch.stack([i, i + 1], dim=1)
    i, j = torch.triu_indices(n_experts, n_experts, offset=1)
    return torch.stack([i, j], dim=1)


# ---------------------------------------------------------------------------
# Hamiltonian assembly + ground state
# ---------------------------------------------------------------------------

def build_hamiltonian(V: torch.Tensor, C: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Real-symmetric H from diagonal energies and off-diagonal couplings.

    V     (..., K)   expert (diabatic) energies
    C     (..., P)   coupling per pair, in the order of `pairs`
    pairs (P, 2)     the (i<j) coupled pairs

    Returns (..., K, K). Built by scatter into a flat view rather than in-place
    indexing so the whole thing stays a single clean autograd node.
    """
    K = V.shape[-1]
    lead = V.shape[:-1]
    flat = torch.diag_embed(V).reshape(*lead, K * K)
    if pairs.numel():
        idx = torch.cat([pairs[:, 0] * K + pairs[:, 1], pairs[:, 1] * K + pairs[:, 0]])
        off = torch.zeros_like(flat).index_add(-1, idx, torch.cat([C, C], dim=-1))
        flat = flat + off
    return flat.reshape(*lead, K, K)


def evb_two_state(V: torch.Tensor, C: torch.Tensor, gap_eps: float = 1e-12):
    """Closed-form ground state of a 2×2 EVB Hamiltonian.

    V (..., 2), C (...,) → (E0 (...,), weights (..., 2), c0 (..., 2)).

    Preferred over the general eigensolver at K=2: it is analytic, so the
    second derivatives that force training needs come out of ordinary autograd
    with no eigenvector-perturbation term to go singular. `gap_eps` floors the
    radicand, which is only reachable when the experts are degenerate *and*
    uncoupled — the one point where dE/dV is genuinely undefined.
    """
    half_sum = 0.5 * (V[..., 0] + V[..., 1])
    d = 0.5 * (V[..., 0] - V[..., 1])
    r = torch.sqrt(d * d + C * C + gap_eps)
    e0 = half_sum - r
    w0 = 0.5 * (1.0 - d / r)
    weights = torch.stack([w0, 1.0 - w0], dim=-1)
    # Relative sign: the eigenvector is ∝ (C, -(d+r)) with d+r ≥ 0, so c0 and c1
    # have opposite signs when C > 0 — which is what makes the cross term
    # 2 c0 c1 C negative and pushes E0 below both experts. `where` (not sign())
    # keeps it ±1 at C = 0, where the vanishing weight makes the choice moot.
    rel = torch.where(C >= 0, -torch.ones_like(C), torch.ones_like(C))
    c0 = torch.stack([weights[..., 0].clamp_min(0).sqrt(),
                      rel * weights[..., 1].clamp_min(0).sqrt()], dim=-1)
    return e0, weights, c0


def evb_ground_state(H: torch.Tensor, gap_eps: float = 1e-12):
    """Lowest eigenvalue, expert weights and eigenvector of a batch of H.

    H (..., K, K) real symmetric → (E0 (...,), weights (..., K), c0 (..., K))
    with weights = c0².

    K=2 takes the closed form above. K>2 goes through `torch.linalg.eigh`;
    the eigenvalue's own gradient is c0 c0ᵀ and needs no gap, but the *second*
    derivative (i.e. training on forces) differentiates the eigenvector and so
    scales as 1/(E1-E0). Nonzero couplings keep that gap open — it is 2|C| at
    K=2 — which is exactly the regime EVB is meant to be used in; a run with all
    couplings driven to zero and two experts crossing is the degenerate case to
    watch for.
    """
    if H.shape[-1] == 1:
        e0 = H[..., 0, 0]
        ones = torch.ones_like(e0).unsqueeze(-1)
        return e0, ones, ones
    if H.shape[-1] == 2:
        V = torch.stack([H[..., 0, 0], H[..., 1, 1]], dim=-1)
        return evb_two_state(V, H[..., 0, 1], gap_eps)
    evals, evecs = torch.linalg.eigh(H)
    e0 = evals[..., 0]
    c0 = evecs[..., :, 0]
    return e0, c0 * c0, c0


def softmin_energy(V: torch.Tensor, tau: torch.Tensor):
    """-τ log Σ_i exp(-V_i/τ) and its softmax weights (a smooth min over experts).

    V (n_seg, K), tau (n_seg,) — one temperature per segment, so a caller can
    make τ extensive (τ ∝ atoms in the segment) and keep the smoothing scale
    comparable across system sizes.
    """
    logits = -V / tau.unsqueeze(-1)
    e = -tau * torch.logsumexp(logits, dim=-1)
    return e, torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Expert-collapse regularisers
# ---------------------------------------------------------------------------

def diversity_loss(weights: torch.Tensor, kind: str = 'load', eps: float = 1e-12):
    """Pressure against expert collapse, from the per-segment expert weights.

    weights (n_seg, K) — rows sum to 1 (EVB: c_0i²; MoE: the softmax gate).

    'load'    the Switch-Transformer load-balancing loss, K·Σ_i f_i·p_i, with
              f_i the fraction of segments where expert i wins and p_i its mean
              weight. Minimised (= 1) by uniform usage across the batch; note
              it pushes on the *batch* marginal, not on per-segment sharpness,
              so experts can still specialise sharply as long as each is used.
    'entropy' mean per-segment entropy minus the entropy of the batch-mean
              weights: rewards *sharp* per-configuration assignments that are
              nonetheless *spread* across the batch — specialisation, which is
              the thing the theory note actually wants from the experts.
    'cv'      coefficient of variation of mean expert usage; the softest of the
              three, and blind to how sharp any individual assignment is.
    """
    if weights.dim() != 2:
        raise ValueError(f"weights must be (n_seg, K), got shape {tuple(weights.shape)}")
    K = weights.shape[-1]
    p = weights.mean(0)
    if kind == 'load':
        top = weights.argmax(-1)
        f = torch.zeros_like(p).index_add(
            0, top, torch.ones_like(top, dtype=weights.dtype)) / weights.shape[0]
        return K * (f * p).sum()
    if kind == 'entropy':
        h_seg = -(weights * (weights + eps).log()).sum(-1).mean()
        h_batch = -(p * (p + eps).log()).sum()
        return h_seg - h_batch
    if kind == 'cv':
        return p.std(unbiased=False) / (p.mean() + eps)
    raise ValueError(f"diversity_loss kind must be 'load', 'entropy' or 'cv', got {kind!r}")


# ---------------------------------------------------------------------------
# The read-out head
# ---------------------------------------------------------------------------


class MixtureReadout(nn.Module):
    """K expert energies + couplings over shared per-edge invariants → one energy.

    Drop-in replacement for ``ECENet._apply_output`` + the atomic-energy sum,
    used when ``ECENet(n_experts>1)``. Structure mirrors the theory note's
    proposed architecture: one shared encoder (the model's equivariant trunk)
    feeds parallel expert heads and coupling heads, and the mixing rule turns
    them into the potential.

    Every head reuses the base read-out's exact per-edge recipe — ``MLP(inv)``
    dotted with the cutoff-enveloped radial basis of the bond length (or a
    single channel times ``f_cut`` when ``n_max_d`` is None) — so expert
    energies and couplings decay smoothly to zero at ``r_cut`` just as the
    single-head energy does, and forces stay continuous. Diabatic energies and
    couplings also carry a per-(type, expert) and per-(type, pair) atomic
    constant, the multi-expert analogue of ``atomic_energy``: it survives when
    an atom has no neighbours inside the cutoff, so the mixture is continuous
    as the last edge leaves.

    The model's own scalar ``atomic_energy`` stays *outside* the mixture. That
    is not an approximation: adding f(R)·I to H shifts every eigenvalue by
    f(R), so a term common to all experts is a gauge choice that can be pulled
    out of the Hamiltonian exactly (theory note §14).

    Args:
        n_types, n_features, n_max_d, r_cut, cutoff_type, hidden_dims,
        activation: mirror the base read-out's configuration
        n_experts:  K, the number of diabatic experts
        mixture:    'evb' | 'moe' | 'softmin' | 'mean' (see module docstring)
        scope:      'atom' (one K×K problem per atom, size-consistent) or
                    'global' (one per structure — the literal theory note)
        coupling:   'mlp' (learned C(R), the default), 'const' (the per-type
                    atomic constant only — H's off-diagonals do not vary with
                    geometry) or 'none' (C ≡ 0, i.e. hard min over experts)
        coupling_topology: 'full' | 'chain' | 'none' — which pairs couple
        coupling_init:     initial per-(type, pair) atomic coupling, eV/atom.
                    Deliberately nonzero: it opens the E1-E0 gap at step 0,
                    which is what keeps the eigen-derivatives well conditioned.
        coupling_positive: pass the assembled coupling through softplus so C > 0
                    (the sign of C picks which adiabatic state is stabilised;
                    only |C| matters at K=2). ``coupling_init`` is then a
                    pre-activation value — softplus(-3) ≈ 0.05.
        expert_init: std of the random per-(type, expert) atomic energies. The
                    symmetry breaker: with identical diagonals *and* weak
                    couplings the Hamiltonian starts degenerate and the experts
                    have no gradient signal to differentiate.
        tau:        softmin temperature in eV/atom (scaled by segment size)
        gap_eps:    radicand floor in the K=2 closed form
    """

    def __init__(self, n_types, n_features, n_max_d, r_cut, cutoff_type,
                 hidden_dims, activation, n_experts,
                 mixture='evb', scope='atom', coupling='mlp',
                 coupling_topology='full', coupling_init=0.05,
                 coupling_positive=False, expert_init=0.05,
                 tau=0.1, gap_eps=1e-12):
        super().__init__()
        # Deferred: model.py imports this module at import time, so the reverse
        # import can only happen once ECENet is actually being constructed.
        from ecenet.model import OutputMLP

        if mixture not in MIXTURES:
            raise ValueError(f"mixture must be one of {MIXTURES}, got {mixture!r}")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
        if coupling not in COUPLINGS:
            raise ValueError(f"coupling must be one of {COUPLINGS}, got {coupling!r}")
        if n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {n_experts}")

        self.n_experts = int(n_experts)
        self.mixture = mixture
        self.scope = scope
        self.coupling = coupling
        self.coupling_positive = bool(coupling_positive)
        self.n_max_d = n_max_d
        self.r_cut = r_cut
        self.cutoff_type = cutoff_type
        self.tau = float(tau)
        self.gap_eps = float(gap_eps)

        pairs = coupling_pairs(self.n_experts, coupling_topology)
        if coupling == 'none':
            pairs = pairs[:0]
        self.register_buffer('pairs', pairs, persistent=False)
        self.n_pairs = pairs.shape[0]

        n_out = n_max_d if n_max_d is not None else 1
        act = {'silu': nn.SiLU, 'tanh': nn.Tanh, 'relu': nn.ReLU,
               'gelu': nn.GELU}.get(activation, nn.SiLU)
        dims = [n_features] + list(hidden_dims)

        # Expert energies. The near-zero last layer of OutputMLP is kept (the
        # per-edge energies start small, as in the single-head read-out) but its
        # rows are independent random draws, so the experts are not clones.
        self.expert_net = OutputMLP(dims + [self.n_experts * n_out], activation=act())
        self.expert_atomic = nn.Parameter(
            torch.randn(n_types, self.n_experts) * expert_init)

        # Couplings: learned C(R) plus a per-(type, pair) constant.
        self.coupling_net = None
        if self.n_pairs:
            if coupling == 'mlp':
                self.coupling_net = OutputMLP(dims + [self.n_pairs * n_out],
                                              activation=act())
            self.coupling_atomic = nn.Parameter(
                torch.full((n_types, self.n_pairs), float(coupling_init)))

        # Ordinary-MoE gate: per-edge logits, envelope-weighted mean over the
        # segment (intensive and smooth at r_cut), plus a learned bias. Zero-init
        # last layer → uniform gate at step 0.
        self.gate_net = None
        if mixture == 'moe':
            self.gate_net = OutputMLP(dims + [self.n_experts], activation=act())
            self.gate_bias = nn.Parameter(torch.zeros(self.n_experts))

    # ── per-edge head → per-segment sum ───────────────────────────────────

    def _edge_values(self, net, invariants, dist_ij, n_blocks):
        """MLP(inv) → (n_edges, n_blocks), with the base read-out's radial recipe."""
        raw = net(invariants)
        if self.n_max_d is not None:
            basis = radial_basis(dist_ij, self.r_cut, self.n_max_d,
                                 cutoff_type=self.cutoff_type)       # (n_e, n_max_d)
            return (raw.reshape(-1, n_blocks, self.n_max_d) * basis.unsqueeze(1)).sum(-1)
        env = get_cutoff_fn(self.cutoff_type)(dist_ij, self.r_cut)
        return raw * env.unsqueeze(-1)

    @staticmethod
    def _segment_sum(values, seg, n_seg):
        out = values.new_zeros(n_seg, values.shape[-1])
        return out.index_add(0, seg, values)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, invariants, dist_ij, types, edge_atom, atom_struct,
                n_atoms, n_struct):
        """Mix the experts into per-structure energies.

        Args:
            invariants:  (n_edges, n_features) per-edge m=0 invariants
            dist_ij:     (n_edges,) bond lengths
            types:       (n_atoms,) atom-type indices
            edge_atom:   (n_edges,) index of the atom each directed edge is
                         centred on (``edge_i``) — the per-atom decomposition
            atom_struct: (n_atoms,) structure index per atom, or None for one
                         structure
            n_atoms, n_struct: sizes

        Returns:
            (energies (n_struct,), info) where info carries the diabatic
            energies, couplings and expert weights per segment — the inputs a
            trainer needs for ``diversity_loss`` and the quantities to log when
            asking which expert owns which chemistry. ``info['c0']`` is the
            *signed* ground-state eigenvector under 'evb' (weights are its
            square; the signs are what make the coupling correction
            Σ_{i≠j} c_i C_ij c_j negative) and None under the other rules,
            which have no eigenvector behind their weights.
        """
        device = invariants.device
        if atom_struct is None:
            atom_struct = torch.zeros(n_atoms, dtype=torch.long, device=device)

        # A "segment" is whatever gets its own K×K Hamiltonian.
        if self.scope == 'atom':
            edge_seg, atom_seg, n_seg = edge_atom, None, n_atoms
            seg_struct = atom_struct
        else:
            edge_seg, atom_seg, n_seg = atom_struct[edge_atom], atom_struct, n_struct
            seg_struct = None

        def to_seg(per_atom):
            if atom_seg is None:      # scope='atom': segment == atom
                return per_atom
            return self._segment_sum(per_atom, atom_seg, n_seg)

        # ── Diabatic energies V_i ────────────────────────────────────────
        V = to_seg(self.expert_atomic[types])                        # (n_seg, K)
        if dist_ij.numel():
            V = V + self._segment_sum(
                self._edge_values(self.expert_net, invariants, dist_ij, self.n_experts),
                edge_seg, n_seg)

        # ── Couplings C_ij ───────────────────────────────────────────────
        C = V.new_zeros(n_seg, self.n_pairs)
        if self.n_pairs:
            C = to_seg(self.coupling_atomic[types])
            if self.coupling_net is not None and dist_ij.numel():
                C = C + self._segment_sum(
                    self._edge_values(self.coupling_net, invariants, dist_ij, self.n_pairs),
                    edge_seg, n_seg)
            if self.coupling_positive:
                C = nn.functional.softplus(C)

        # ── Mix ──────────────────────────────────────────────────────────
        c0 = None
        if self.mixture == 'evb':
            H = build_hamiltonian(V, C, self.pairs)
            e_seg, weights, c0 = evb_ground_state(H, self.gap_eps)
        elif self.mixture == 'mean':
            weights = V.new_full(V.shape, 1.0 / self.n_experts)
            e_seg = V.mean(-1)
        elif self.mixture == 'softmin':
            # τ per segment ∝ its atom count: an extensive energy needs an
            # extensive temperature, or the smoothing vanishes as N grows and
            # softmin silently becomes a hard min.
            n_at = to_seg(torch.ones(n_atoms, 1, dtype=V.dtype, device=device))
            e_seg, weights = softmin_energy(V, self.tau * n_at.squeeze(-1).clamp_min(1.0))
        else:                                   # 'moe' — ordinary softmax gate
            logits = self.gate_bias.expand(n_seg, self.n_experts)
            if self.gate_net is not None and dist_ij.numel():
                env = get_cutoff_fn(self.cutoff_type)(dist_ij, self.r_cut).unsqueeze(-1)
                num = self._segment_sum(self.gate_net(invariants) * env, edge_seg, n_seg)
                den = self._segment_sum(env, edge_seg, n_seg).clamp_min(1e-12)
                logits = logits + num / den
            weights = torch.softmax(logits, dim=-1)
            e_seg = (weights * V).sum(-1)

        energies = (e_seg if seg_struct is None
                    else e_seg.new_zeros(n_struct).index_add(0, seg_struct, e_seg))
        info = {'expert_energies': V, 'coupling': C, 'weights': weights,
                'c0': c0, 'energy_seg': e_seg, 'seg_struct': seg_struct}
        return energies, info

    def extra_repr(self):
        return (f"n_experts={self.n_experts}, mixture={self.mixture!r}, "
                f"scope={self.scope!r}, coupling={self.coupling!r}, "
                f"n_pairs={self.n_pairs}")
