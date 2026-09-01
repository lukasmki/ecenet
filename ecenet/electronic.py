"""ecenet/electronic.py — total-charge / total-spin conditioning.

ECENet is otherwise a pure function of geometry and composition: an anion and
its neutral parent, frozen at the same geometry, are the same input and get the
same energy. This module supplies the missing variable — the **global
electronic state** — as a small invariant feature vector that conditions the
model, wired in via ``ECENet(charge_spin=True)`` and passed at call time as
``forward(..., total_charge=, total_spin=)``.

State vector
------------
`state_features` maps a structure's total charge `Q` (in units of e, so a
cation is `+1`) and total spin `S` (the number of *unpaired electrons*,
`S = multiplicity - 1 = 2·S_z`) to

    [ Q, S, Q/N, S/N ]

with `N` the atom count. Extensive and intensive halves are both supplied, on
purpose:

- `Q/N`, `S/N` are what a **size-consistent** conditioning needs. ECENet's
  energy is a sum of local terms; a per-atom conditioning signal that is
  invariant under replicating the system (with its charge) is the only kind
  that leaves the energy extensive. Two isolated copies of a `+1` ion see the
  same `Q/N` as one copy does.
- `Q`, `S` are kept alongside because the physical response to adding *one*
  electron is emphatically not intensive: the electron affinity of a 10-atom
  cluster is not a tenth of a 100-atom one's, and a model given only `Q/N`
  cannot tell `+1` on 10 atoms from `+10` on 100.

All four numbers are invariant under rotation, translation and permutation, so
they can be mixed into any *invariant* slot of the model without touching
equivariance. Where they enter matters — see below — but not what they are.

Where it enters
---------------
Two sites, each optional and each **identity at init**, so turning the
conditioning on does not perturb a model at step 0:

``charge_spin_film``
    A FiLM gate on the freshly built edge features, in the bond frame, right
    beside the element gate of `ecenet/film.py` (it *is* an `ElementFiLM`, with
    the state vector as its extra conditioning leg instead of a radial one).
    This is the load-bearing site: it sits in front of the whole equivariant
    stack, so everything downstream — per-edge energies, forces, the LES latent
    charges, the mixture-of-experts invariants — becomes state-dependent. The
    equivariance argument is unchanged from the element gate, because the state
    vector is an invariant scalar: γ scales `cos`/`sin` of a given `(c, m)`
    together, and β touches `m = 0` only.

``charge_spin_atomic``
    `StateAtomicEnergy`: a per-atom scalar read off the atom's element and the
    state vector — the state-conditioned analogue of ECENet's `atomic_energy`
    baseline. It carries the part of the response that is not geometric (the
    cost of putting this charge on this composition at all), and, needing no
    edges, it is the only term that survives for an isolated ion whose
    neighbours have all left `r_cut_edge`. Without it a lone Na⁺ and a lone Na
    would be exactly degenerate.

What this is *not*
------------------
This is global conditioning, not charge equilibration: nothing here redistributes
charge between atoms or enforces `Σ_i q_i = Q` on the LES latent charges (which
are latent, not physical, and carry an arbitrary global sign). A charged
system's long-range physics still comes from LES if it is enabled.
"""

import torch
import torch.nn as nn

# [Q, S, Q/N, S/N] — the width of the conditioning leg every consumer sizes to.
N_STATE_FEATURES = 4


def _as_vector(x, n_struct, device, dtype):
    """A per-structure scalar input → an (n_struct,) tensor.

    Accepts None (→ zeros, i.e. neutral / closed shell), a python number, a
    0-dim tensor (broadcast to every structure), or anything tensor-like of
    shape (n_struct,). Anything else is a shape error, raised here rather than
    broadcast into silence downstream.
    """
    if x is None:
        return torch.zeros(n_struct, device=device, dtype=dtype)
    t = torch.as_tensor(x, device=device, dtype=dtype)
    if t.dim() == 0:
        return t.expand(n_struct)
    t = t.reshape(-1)
    if t.shape[0] == 1:
        return t.expand(n_struct)
    if t.shape[0] != n_struct:
        raise ValueError(
            f"expected a scalar or {n_struct} per-structure values, got "
            f"{t.shape[0]}")
    return t


def is_nonzero(x):
    """True if a charge/spin input is anything but None-or-all-zero.

    Used to tell a *neutral closed-shell* default (which every caller passes
    implicitly, and which a state-blind model handles correctly) from a real
    charge or spin (which it does not) — see ECENet._state_features.
    """
    if x is None:
        return False
    if isinstance(x, torch.Tensor):
        return bool((x != 0).any())
    if isinstance(x, (list, tuple)):
        return any(is_nonzero(v) for v in x)
    return bool(x != 0)


def state_features(total_charge, total_spin, atom_counts, device, dtype):
    """[Q, S, Q/N, S/N] per structure — see the module docstring.

    Args:
        total_charge: total charge in e; None, a scalar, or (B,)
        total_spin:   unpaired electrons (multiplicity - 1); None, scalar, or (B,)
        atom_counts:  (B,) atom count per structure (any array-like)
        device, dtype: of the returned tensor

    Returns:
        (B, N_STATE_FEATURES)
    """
    n = torch.as_tensor(atom_counts, device=device, dtype=dtype).reshape(-1)
    B = n.shape[0]
    q = _as_vector(total_charge, B, device, dtype)
    s = _as_vector(total_spin, B, device, dtype)
    n = n.clamp_min(1.0)                       # an empty structure gets Q/N = Q
    return torch.stack([q, s, q / n, s / n], dim=-1)


class StateAtomicEnergy(nn.Module):
    """Per-atom energy from the atom's element and the global electronic state.

    ``MLP([ embed(Z_i) ; Q, S, Q/N, S/N ]) → ℝ``, summed over atoms alongside
    ECENet's `atomic_energy`. Zero-init last layer, so it contributes exactly 0
    at init and the model starts wherever it started without the conditioning.

    At `Q = S = 0` this is *not* forced to zero after training — it is simply a
    state-conditioned atomic baseline, and the neutral state is one of the
    states it fits. That is the intended behaviour: it absorbs per-element
    ionisation/affinity-like offsets, which are only meaningful as differences
    between states of the same composition.

    Args:
        n_types:   number of element types
        embed_dim: width of the element embedding (default 16)
        hidden:    hidden width(s): int, list/tuple, or None → [max(4*P, 32)]
    """

    def __init__(self, n_types, embed_dim=16, hidden=None):
        super().__init__()
        P = int(embed_dim)
        self.embed = nn.Embedding(n_types, P)
        if hidden is None:
            hidden = [max(4 * P, 32)]
        elif isinstance(hidden, (list, tuple)):
            hidden = [int(h) for h in hidden]
        else:
            hidden = [int(hidden)]
        dims = [P + N_STATE_FEATURES] + hidden + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, types, state_atom):
        """types (N,), state_atom (N, N_STATE_FEATURES) → (N,) atomic energies."""
        if state_atom.shape[-1] != N_STATE_FEATURES:
            raise ValueError(
                f"state_atom must be (N, {N_STATE_FEATURES}), got "
                f"{tuple(state_atom.shape)}")
        x = torch.cat([self.embed(types), state_atom.to(self.embed.weight.dtype)],
                      dim=-1)
        return self.mlp(x).squeeze(-1)

    def extra_repr(self):
        return f"n_state_features={N_STATE_FEATURES}"
