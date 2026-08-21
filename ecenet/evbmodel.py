"""ecenet/evbmodel.py — MultiECENet: EVB state mixing over ECENet trunks.

A single ECENet fits one potential energy surface. That is the wrong shape for
chemistry where the electronic character *changes* along the reaction coordinate
— a proton transferring between two bases, a charge hopping between fragments, a
bond homolysing from closed-shell to open-shell. The adiabatic ground state there
has a cusp-like avoided crossing that a single smooth network has to memorise.

MultiECENet instead learns several *diabatic* states — each a smooth, well-behaved
surface of fixed charge/spin character — and recovers the physical ground state by
empirical-valence-bond mixing: build the S×S Hamiltonian

    H[k, k] = E_k(geometry)     diabat energy  (+ per-state atomic baseline)
    H[k, l] = V_kl(geometry)    diabatic coupling, k != l

and take its lowest eigenvalue. State switching is then an eigenvector rotation
that happens on the fly, and the surfaces the networks actually fit stay smooth.

Layout::

                         ┌── head (0,0) ──► H₀₀
    positions, types ──► │── head (1,1) ──► H₁₁         E = eigvalsh(H)[0]
       ECENet trunk      │── head (0,1) ──► H₀₁   ──►   c = ground eigenvector
       (per-edge         └── ...                        w = c²  (state weights)
        invariants)               H (S×S, symmetric)

Every matrix element is a sum of smooth per-edge terms sharing the trunk's radial
/ cutoff-envelope readout convention, so it decays to zero at ``r_cut_edge``
exactly as an ECENet energy does.

Known limitation: the EVB matrix is *global* — one S×S per structure — so the
mixing is a whole-system decision and the model is not size-extensive. That is
standard for EVB and appropriate for a single reaction centre; many independent
reactive sites in one box would need a local / per-fragment EVB instead.
"""

import torch
from torch import nn

from ecenet.model import ECENet, OutputMLP


class MultiECENet(nn.Module):
    """Committee of ECENet diabats mixed by an empirical valence bond matrix.

    Args:
        n_types:      number of atom types
        states:       one (charge, multiplicity) pair per diabat. These define
                      the *sectors*: a structure declaring charge Q (and spin M)
                      mixes only the diabats whose label matches, because a
                      resonance structure conserves total charge and spin. Give a
                      sector two or more diabats for it to mix at all; a
                      one-diabat sector is a plain single-surface model for that
                      charge. Passing no charge/spin to a forward leaves every
                      diabat active, which is the ``S``-state model with no
                      sectoring.
        shared_trunk: True  — one ECENet produces the per-edge invariants and
                              every head reads them (cheapest; the states share
                              a single representation).
                      False — one ECENet per diabat. Head (k,k) reads trunk k;
                              off-diagonal head (k,l) reads trunk k's and trunk
                              l's invariants concatenated.
        mix_mode:     'eigvalsh' (default) — E = eigvalsh(H)[0]. Its backward is
                      the division-free ``V diag(g) Vᵀ``, so first derivatives
                      stay well conditioned near an avoided crossing.
                      'eigvector' — the explicit EVB combination E = cᵀHc with c
                      the detached ground eigenvector. Same energy, and exact
                      forces by Hellmann–Feynman, but its *second* derivatives
                      drop the eigenvector-response term
                      ``Σ_{j≠0} 2|c_jᵀ dH c_0|² / (λ₀ − λ_j)``, which biases the
                      parameter gradients of a force loss. Prefer the default for
                      training; 'eigvector' is for analysis and for reproducing
                      textbook EVB arithmetic exactly.
        **ecenet_kwargs: forwarded verbatim to every ``ECENet`` trunk (l_max,
                      n_max, embed_dim, n_layers, n_mp, element_film, ...).

    Entry points mirror ECENet one-for-one — ``forward``, ``forward_pbc``,
    ``forward_batch_multi``, ``forward_batch`` — with the same signatures and
    return shapes, so an EVB model drops in wherever an ECENet went.
    """

    def __init__(self, n_types: int, states=((0, 1), (0, 1)),
                 shared_trunk: bool = True, mix_mode: str = 'eigvalsh',
                 **ecenet_kwargs):
        super().__init__()
        states = tuple(tuple(int(x) for x in s) for s in states)
        if len(states) < 1:
            raise ValueError("need at least one diabatic state")
        if any(len(s) != 2 for s in states):
            raise ValueError(f"each state must be a (charge, multiplicity) pair, got {states!r}")
        if mix_mode not in ('eigvalsh', 'eigvector'):
            raise ValueError(f"mix_mode must be 'eigvalsh' or 'eigvector', got {mix_mode!r}")

        self.n_types = n_types
        self.states = states
        self.n_states = len(states)
        self.shared_trunk = bool(shared_trunk)
        self.mix_mode = mix_mode
        # A buffer, not a plain attribute, so the charge/spin assignment travels
        # with the state_dict — an EVB checkpoint is meaningless without it.
        self.register_buffer('state_labels', torch.tensor(states, dtype=torch.long))
        # (charge, multiplicity) -> diabat indices, for error messages and for
        # callers that want to know what the model can actually describe.
        self.sectors = {}
        for k, s in enumerate(states):
            self.sectors.setdefault(s, []).append(k)

        S = self.n_states
        self.trunks = nn.ModuleList([
            ECENet(n_types=n_types, **ecenet_kwargs)
            for _ in range(1 if self.shared_trunk else S)])

        # ── EVB heads: one readout per upper-triangle matrix element ────────
        # Dims and activation are read off the trunk's own output_net rather than
        # re-derived from kwargs, so the heads track ECENet's readout config
        # (output_hidden_dims, n_max_d) automatically.
        ref = self.trunks[0].output_net
        hidden = [lin.out_features for lin in ref.linears[:-1]]
        n_out = ref.linears[-1].out_features
        act_cls = type(ref.activation)
        n_feat = self.trunks[0].n_features_per_m

        self.heads = nn.ModuleDict()
        for k in range(S):
            for l in range(k, S):
                # Off-diagonal heads see both trunks' invariants when the trunks
                # are independent; with a shared trunk there is only one view.
                in_dim = n_feat if (k == l or self.shared_trunk) else 2 * n_feat
                # Diagonal heads keep ECENet's zero-init-last (diabat energies
                # start at the atomic baseline). Off-diagonal heads deliberately
                # do NOT: zero couplings with equal diagonals make H a multiple
                # of the identity, whose ground eigenvector — and so the
                # eigenvalue gradient — is arbitrary within the degenerate
                # subspace. Small nonzero couplings lift that from step 0.
                self.heads[f'{k}_{l}'] = OutputMLP(
                    [in_dim] + hidden + [n_out], activation=act_cls(),
                    zero_init_last=(k == l))

        # Per-state, per-type atomic energy baseline. Diabats of different
        # charge/spin differ by eV-scale absolute offsets, so ECENet's single
        # shared baseline cannot serve them all. Off-diagonal elements get no
        # baseline: a coupling must vanish where the states do not interact, and
        # a per-type constant would instead grow with system size.
        self.atomic_energy = nn.Parameter(torch.zeros(S, n_types))

    # ── Trunk attribute pass-through ────────────────────────────────────────
    # ECENetCalculator reads r_cut_edge / r_cut_neighbor off the model and sets
    # analytic_ace_basis on it; forwarding to the trunks keeps that working.

    @property
    def r_cut_edge(self):
        return self.trunks[0].r_cut_edge

    @property
    def r_cut_neighbor(self):
        return self.trunks[0].r_cut_neighbor

    @property
    def analytic_ace_basis(self):
        return self.trunks[0].analytic_ace_basis

    @analytic_ace_basis.setter
    def analytic_ace_basis(self, value):
        for trunk in self.trunks:
            trunk.analytic_ace_basis = value

    # ── EVB matrix assembly ─────────────────────────────────────────────────

    def _run_trunks(self, run):
        """Run every trunk under capture, discarding its own energy.

        ``run(trunk)`` performs one full trunk forward; the capture hook in
        ``ECENet._apply_output`` hands back that pass's per-edge invariants. The
        trunk's own readout still fires and its result is thrown away — one extra
        MLP over the edges, negligible beside the equivariant stack.
        """
        caps = []
        for trunk in self.trunks:
            with trunk.capture_edges() as cap:
                run(trunk)
            caps.append(cap)
        return caps

    # Inactive diabats are pushed to this diagonal energy (model units, eV) and
    # decoupled, so they can never be the ground state. Large enough to dominate
    # any physical energy after reference subtraction, small enough to stay exact
    # in float32. The masked matrix is block-diagonal, so eigvalsh resolves the
    # active block without conditioning trouble.
    _MASKED_DIAG = 1.0e6

    def _active_mask(self, charge, spin, device, n_struct=1):
        """Which diabats a structure may mix: (n_struct, S) bool, or None.

        ``charge`` / ``spin`` are None, a scalar, or a per-structure sequence.
        None on an axis means "don't filter on it" — passing charge alone
        selects every multiplicity carrying that charge, which is what a
        closed-shell dataset (SPICE: all singlets) wants.
        """
        if charge is None and spin is None:
            return None
        labels = self.state_labels                                  # (S, 2)
        mask = torch.ones(n_struct, self.n_states, dtype=torch.bool, device=device)
        for axis, val in ((0, charge), (1, spin)):
            if val is None:
                continue
            v = torch.as_tensor(val, dtype=torch.long, device=device).reshape(-1)
            if v.numel() == 1:
                v = v.expand(n_struct)
            elif v.numel() != n_struct:
                raise ValueError(f"expected 1 or {n_struct} values, got {v.numel()}")
            mask = mask & (labels[:, axis].unsqueeze(0) == v.unsqueeze(1))
        if not mask.any(dim=1).all():
            bad = (~mask.any(dim=1)).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(
                f"no diabat matches the requested sector for structure(s) {bad[:5]}"
                f"{'...' if len(bad) > 5 else ''} (charge={charge!r}, spin={spin!r}). "
                f"Model sectors (charge, multiplicity): {sorted(self.sectors)}")
        return mask

    def _apply_mask(self, H, active):
        """Decouple and lift the inactive diabats out of the ground state.

        Zeroing an inactive state's couplings makes its diagonal an isolated
        eigenvalue, and lifting that diagonal far above the active block means
        eigvalsh(H)[0] is exactly the ground state of the sector, with gradients
        w.r.t. every inactive entry identically zero. Masking (rather than
        slicing) keeps the matrix a uniform (..., S, S) across a batch whose
        structures sit in different sectors, so one batched eigvalsh serves all.

        The lifted diagonals are spread (k+1)·_MASKED_DIAG rather than sharing
        one value, and that spread is load-bearing: identical lifts would make
        the masked states exactly degenerate, and eigvalsh's *second* derivative
        carries 1/(λi − λj) terms that blow up to NaN there. Force training
        backpropagates through the force, so it hits that double backward on
        every step. Spacing the lifts by _MASKED_DIAG keeps those denominators
        at ~1e6, i.e. harmless.
        """
        if active is None:
            return H
        active = active.reshape(H.shape[:-2] + (self.n_states,))
        H = torch.where(active.unsqueeze(-1) & active.unsqueeze(-2),
                        H, H.new_zeros(()))
        spread = self._MASKED_DIAG * torch.arange(
            1, self.n_states + 1, device=H.device, dtype=H.dtype)
        lift = torch.where(active, H.new_zeros(()), spread.expand(active.shape))
        return H + torch.diag_embed(lift)

    def _assemble(self, caps, baseline, reduce_edges):
        """Build the symmetric EVB matrix from the captured edge features.

        Args:
            caps:         per-trunk capture dicts from _run_trunks
            baseline:     (..., S) per-state atomic baseline, already reduced
            reduce_edges: (n_edges,) per-edge values → (...) per-structure scalar

        Returns:
            H: (..., S, S), symmetric by construction
        """
        S = self.n_states
        # A structure with no edges never reaches _apply_output, so nothing was
        # captured: every element is just its baseline, couplings zero.
        no_edges = 'invariants' not in caps[0]

        vals = {}
        for k in range(S):
            for l in range(k, S):
                if no_edges:
                    v = baseline.new_zeros(baseline.shape[:-1])
                else:
                    src = caps[0] if self.shared_trunk else caps[k]
                    trunk = self.trunks[0] if self.shared_trunk else self.trunks[k]
                    inv = src['invariants']
                    if k != l and not self.shared_trunk:
                        inv = torch.cat([inv, caps[l]['invariants']], dim=-1)
                    per_edge = trunk._apply_output(inv, src['dist_ij'],
                                                   net=self.heads[f'{k}_{l}'])
                    v = reduce_edges(per_edge)
                if k == l:
                    v = v + baseline[..., k]
                vals[(k, l)] = v

        # Stack rather than index-assign into a zeros tensor: no in-place writes
        # in the autograd graph, and symmetry holds because (k,l) and (l,k) pull
        # the identical tensor. Works unchanged for scalar or (B,) elements.
        rows = [torch.stack([vals[(min(k, l), max(k, l))] for l in range(S)], dim=-1)
                for k in range(S)]
        return torch.stack(rows, dim=-2)

    def _baseline(self, types):
        """(S,) per-state atomic-energy baseline for one structure."""
        return self.atomic_energy[:, types].sum(-1)

    def ground_vector(self, H):
        """Detached ground-state eigenvector c of H: (..., S).

        The sign is arbitrary (an eigenvector is defined up to sign); c enters
        every EVB expression quadratically, so nothing downstream depends on it.
        """
        with torch.no_grad():
            return torch.linalg.eigh(H.detach())[1][..., :, 0]

    def _ground_state(self, H):
        """Adiabatic ground-state energy from the EVB matrix."""
        if self.mix_mode == 'eigvalsh':
            return torch.linalg.eigvalsh(H)[..., 0]
        c = self.ground_vector(H)
        return torch.einsum('...k,...kl,...l->...', c, H, c)

    # ── Inspection ──────────────────────────────────────────────────────────

    def evb_matrix(self, *args, **kwargs):
        """The EVB matrix H — same arguments as ``forward``."""
        return self.forward(*args, return_matrix=True, **kwargs)[1]

    def state_weights(self, *args, **kwargs):
        """Ground-state diabatic populations c²: (..., S), summing to 1.

        The practical read-out of "which state is the system in right now":
        w[k] ≈ 1 means diabat k dominates; an even split means the geometry sits
        in the avoided-crossing region.
        """
        return self.ground_vector(self.evb_matrix(*args, **kwargs)) ** 2

    # ── Forward paths (mirroring ECENet) ────────────────────────────────────

    @staticmethod
    def _reject_embeddings(return_embeddings):
        if return_embeddings:
            raise NotImplementedError(
                "MultiECENet does not return per-atom embeddings: the trunks are "
                "diabatic, so there is no single embedding for the mixed ground "
                "state. Read a trunk directly (model.trunks[k]) if you need the "
                "per-state ones.")

    def forward(self, positions, types, charge=None, spin=None,
                return_embeddings=False, return_matrix=False):
        """Adiabatic ground-state energy.

        Args:
            positions:     (n_atoms, 3)
            types:         (n_atoms,) atom-type indices
            charge:        total charge; restricts mixing to that sector.
                           None leaves every diabat active.
            spin:          multiplicity; restricts further. None does not filter.
            return_matrix: also return the (S, S) EVB matrix

        Returns:
            energy: scalar tensor — or (energy, H) when return_matrix
        """
        self._reject_embeddings(return_embeddings)
        caps = self._run_trunks(lambda t: t.forward(positions, types))
        H = self._assemble(caps, self._baseline(types), lambda pe: pe.sum())
        active = self._active_mask(charge, spin, H.device)
        H = self._apply_mask(H, None if active is None else active[0])
        energy = self._ground_state(H)
        return (energy, H) if return_matrix else energy

    def forward_pbc(self, positions, types, edge_i, edge_j, shift_vecs_edge,
                    nb_src, nb_dst, shift_vecs_nb, charge=None, spin=None,
                    return_embeddings=False, return_matrix=False):
        """Ground-state energy under periodic boundary conditions.

        Arguments match ``ECENet.forward_pbc``; the precomputed topology is
        shared by every trunk.
        """
        self._reject_embeddings(return_embeddings)
        caps = self._run_trunks(lambda t: t.forward_pbc(
            positions, types, edge_i, edge_j, shift_vecs_edge,
            nb_src, nb_dst, shift_vecs_nb))
        H = self._assemble(caps, self._baseline(types), lambda pe: pe.sum())
        active = self._active_mask(charge, spin, H.device)
        H = self._apply_mask(H, None if active is None else active[0])
        energy = self._ground_state(H)
        return (energy, H) if return_matrix else energy

    def forward_batch_multi(self, positions_list, types_list,
                            charge=None, spin=None,
                            return_embeddings=False, topology=None,
                            return_matrix=False):
        """Batched ground-state energies for variable-size structures: (B,).

        With independent trunks the topology is built once up front and reused,
        so the trunks do not each pay for the neighbour search.
        """
        self._reject_embeddings(return_embeddings)
        B = len(positions_list)
        if topology is None and not self.shared_trunk:
            topology = self.trunks[0].build_topology(positions_list)

        caps = self._run_trunks(lambda t: t.forward_batch_multi(
            positions_list, types_list, topology=topology))

        baseline = torch.stack([self._baseline(t) for t in types_list])   # (B, S)
        struct_idx = caps[0].get('struct_idx')
        if struct_idx is None:            # no structure in the batch has edges
            def reduce_edges(pe):
                return baseline.new_zeros(B)
        else:
            def reduce_edges(pe):
                return torch.zeros(B, dtype=pe.dtype, device=pe.device).scatter_add(
                    0, struct_idx, pe)

        H = self._assemble(caps, baseline, reduce_edges)
        H = self._apply_mask(H, self._active_mask(charge, spin, H.device, B))
        energies = self._ground_state(H)
        return (energies, H) if return_matrix else energies

    def forward_batch(self, positions_list, types, topology=None,
                      charge=None, spin=None,
                      return_embeddings=False, return_matrix=False):
        """Batched ground-state energies for structures sharing atom types: (B,).

        With a fixed-topology dict this takes ECENet's vectorized path; otherwise
        it falls through to ``forward_batch_multi``, exactly as ECENet does.
        """
        self._reject_embeddings(return_embeddings)
        B = len(positions_list)
        if not isinstance(topology, dict):
            return self.forward_batch_multi(
                positions_list, [types] * B, charge=charge, spin=spin,
                return_matrix=return_matrix)

        caps = self._run_trunks(lambda t: t.forward_batch(
            positions_list, types, topology=topology))

        baseline = self._baseline(types).expand(B, self.n_states)          # (B, S)
        n_edges = caps[0].get('n_edges_per_struct', 0)

        def reduce_edges(pe):
            return pe.reshape(B, n_edges).sum(dim=1)

        H = self._assemble(caps, baseline, reduce_edges)
        H = self._apply_mask(H, self._active_mask(charge, spin, H.device, B))
        energies = self._ground_state(H)
        return (energies, H) if return_matrix else energies

    def extra_repr(self):
        return (f"n_states={self.n_states}, states={self.states}, "
                f"shared_trunk={self.shared_trunk}, mix_mode={self.mix_mode!r}")
