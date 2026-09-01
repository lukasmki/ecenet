# ECENet Equation Reference

A stage-by-stage specification of the model in `ecenet/model.py`: what each
step computes, on what shapes, and why it is equivariant. The README covers
*usage* (flags, training, checkpoints); this document covers *definitions*.

Everything below describes the code as written. Where a stage has more than one
implementation (analytic vs. autograd ACE basis, fused vs. unfused kernels) the
variants are numerically equivalent and only one is given.

---

## 0. Notation

### Indices

| Symbol | Range | Meaning |
| --- | --- | --- |
| `i`, `j` | `1..N` | atoms |
| `k` | — | a neighbour of atom `i` within `r_cut_neighbor` |
| `e = (i→j)` | `1..E` | a **directed** edge; both `i→j` and `j→i` are present |
| `t`, `Z_i` | `0..n_types-1` | element type |
| `n` | `1..n_max` | radial-basis index |
| `d` | `1..n_max_d` | read-out radial-basis index |
| `l` | `0..l_max` | degree (angular momentum) |
| `m` | `-l..l` (features: `0..m_max`) | order (azimuthal mode) |
| `s = l² + l + m` | `0..n_sph-1` | flat spherical-harmonic slot |
| `c` | `0..F-1` | feature channel |
| `b` | `1..B` | structure in a batch |

### Sizes

```
n_sph = (l_max + 1)²                    spherical-harmonic slots
n_angular = m_max + 1                   retained angular modes (m ≥ 0)
n_base = 2·embed_dim                    channels before the l-expansion
F = n_features_per_m = 2·embed_dim·(l_max + 1)     working feature width
```

`m_max` defaults to `l_max`. The factor 2 in `n_base` is the two **sides** of an
edge — the features of atom `i` and of atom `j` are carried side by side.

### Conventions

- Real spherical harmonics, orthonormal (e3nn convention), from `sphericart`.
- Bond frame: the rotation `D(r̂_ij)` takes the bond direction `r̂_ij` to `ẑ`.
- `f_cut` is a cutoff envelope with `f_cut(0) = 1`, `f_cut(r_c) = 0`, `f_cut'(r_c) = 0`.
- Energies in eV, distances in Å.
- A "segment" (§11) is whatever gets its own mixture problem: an atom or a structure.

---

## 1. Geometry and cutoffs

ECENet uses **two** cutoffs. The larger one selects which pairs become edges;
the smaller one selects which atoms enter each atom's ACE descriptor.

```
E    = { (i, j) : 0 < |r_ij| < r_cut_edge }        directed, both orientations
N(i) = { k     : 0 < |r_ik| < r_cut_neighbor }     ACE neighbour list
```

Per directed edge, with PBC shift `S_e` (zero for isolated systems):

```
d_ij = r_j - r_i + S_e
r_ij = ‖d_ij‖                      (computed as √(‖d‖² + 1e-30) for a finite gradient at 0)
r̂_ij = d_ij / r_ij
```

Topology is **non-differentiable** — built under `no_grad` (`_local_topology`)
and frozen across a strain perturbation. Continuity as an atom crosses either
cutoff is carried entirely by the envelopes in §2, not by the index sets.

`ecenet/radial.py`: `find_edges`, `torch_neighbor_list`, `min_perpendicular_width`.

---

## 2. Radial basis and cutoff envelopes

### Envelopes

```
cosine (C¹):   f_cut(r) = ½(1 + cos(π r / r_c))·𝟙[r < r_c]
poly   (C²):   f_cut(r) = (1 - 10x³ + 15x⁴ - 6x⁵)·𝟙[r < r_c],   x = r/r_c
```

Both satisfy `f_cut(r_c) = f_cut'(r_c) = 0`, so energies and forces are
continuous at the cutoff. `'poly'` additionally has `f_cut''(r_c) = 0`, which
matters when training on forces (the loss differentiates the model twice).

### Basis

```
f_n(r) = √(2/r_c) · sinc(n π r / r_c) · f_cut(r),      n = 1 … n_max
```

with `sinc(x) = sin(x)/x` and `sinc(0) = 1`. Every basis function is enveloped,
so every downstream quantity built from `f_n` inherits the smooth decay to zero.
`radial_basis_with_deriv` returns `df_n/dr` analytically for the ACE backward.

---

## 3. ACE atomic basis

Per atom, an element-resolved density expansion over its neighbours:

```
A[i, t, n, s] = Σ_{k ∈ N(i)}  f_n(r_ik) · Y_s(r̂_ik) · δ(Z_k = t)
```

Shape `(N, n_types, n_max, n_sph)`. This is a one-particle (body-order 2) basis:
higher body order is produced later by the nonlinear layers, not here.

`ACEBasisAnalytic` (`ecenet/ace_basis.py`) implements this as a custom
`autograd.Function`. Its backward uses the analytic spherical-harmonic Jacobian
`∂Y_s/∂x` from `sphericart`, stored as a **plain tensor with no `grad_fn`**:

```
∂L/∂d_ik = Σ_s (∂L/∂Y_s)·(∂Y_s/∂d_ik)  +  (d_ik/r_ik)·Σ_n (∂L/∂f_n)·(df_n/dr)
```

Because the Jacobian carries no graph, the second-order backward needed for
force training never re-traverses the spherical-harmonic graph — the single
largest cost in a naive implementation.

---

## 4. Element-resolved embedding

A per-central-element contraction over neighbour type and radial index:

```
A_emb[i, c, s] = Σ_{t, n} A[i, t, n, s] · W[Z_i, t, n, c]
```

Shape `(N, embed_dim, n_sph)`. `W` has shape `(n_types, n_types, n_max,
embed_dim)`, initialised `𝒩(0, 1/(n_types·n_max))`. The contraction is over the
`(t, n)` axes only, never over `s`, so the `l`-block structure survives intact
and the result rotates exactly as `A` does.

---

## 5. Bond frames and Wigner-D

Node features live in the global frame; every edge then rotates them into its
own bond frame, so that all subsequent operations only need SO(2) equivariance
about the bond axis.

Gather both endpoints and rotate:

```
Ã_e = [ A_emb[i(e)] ; A_emb[j(e)] ]  ∈ ℝ^{n_base × n_sph}       (channel-concat)
A_rot[e, c, s'] = Σ_s Ã_e[c, s] · D_block(r̂_e)[s, s']
```

`D_block` is block-diagonal, one `(2l+1)×(2l+1)` block `D^l` per degree.

### Building `D^1`

`D^1` comes straight from `r̂` by Gram–Schmidt against a fixed reference vector,
with **two charts** so no denominator can vanish (`build_D1_from_rhat`):

```
chart A  (|r_x| < 0.9, ref = x̂):   s_x = √(r_y² + r_z²) ≥ 0.436
chart B  (|r_x| ≥ 0.9, ref = ŷ):   s_y = √(r_x² + r_z²) ≥ 0.9
```

Both denominators are bounded away from zero on their domain, so there is no
singularity and no gradient blow-up anywhere on the sphere. Each chart uses the
double-`where` safe-sqrt idiom so that even the *discarded* branch has a finite
backward — otherwise an edge exactly on an axis (a PBC self-image edge in a
crystal) would produce `NaN` through the dead branch.

The real-SH basis order for `l=1` is `(y, z, x)`.

### Higher `l`

By Clebsch–Gordan recursion from `D^1` (no Euler angles, no trigonometry):

```
D^l = Cᵀ · (D^1 ⊗ D^{l-1}) · C
```

with `C = _real_cg(1, l-1, l)` the real CG coefficients, precomputed once in
`complex128` and cached (`ecenet/spherical.py`).

---

## 6. SO(2) angular representation

In the bond frame the residual symmetry is rotation by `φ` about `ẑ`, under
which the order-`m` component picks up `e^{imφ}`. Features are therefore carried
as **cos/sin Fourier pairs** over `m ≥ 0`.

`SphToAngular` folds the degree `l` into the channel axis and splits `±m`:

```
channel c ↔ (side, embed, l),   c = ((side·embed_dim) + embed)·(l_max+1) + l

A_cos[e, c, m] = A_rot[e, (side,embed), l² + l + m] · 𝟙[m ≤ l]
A_sin[e, c, m] = A_rot[e, (side,embed), l² + l - m] · 𝟙[1 ≤ m ≤ l]
```

Shapes `(E, F, n_angular)`. Two structural facts follow from the masks and are
relied on throughout:

- `A_sin[·, ·, 0] ≡ 0` — `sin(0·φ) = 0` is not a degree of freedom.
- `A_cos[·, c, m] = A_sin[·, c, m] = 0` whenever `m > l(c)` — a triangular
  pattern of **structural zeros**. Anything applied per-`m` must not disturb
  them differentially (see §7).

Modes with `m > m_max` are discarded here; the ACE basis always runs at full
`l_max`, so lowering `m_max` cuts layer cost without truncating the descriptor.

---

## 7. FiLM gates (optional)

Both gates below modulate the freshly built edge features, in the bond frame,
before the layer stack. They differ only in what conditions them.

### Element gate (`element_film=True`)

One element(+distance)-conditioned modulation:

```
g_e = MLP( [ a(Z_i) ; b(Z_j) ; φ(r_ij) ] )        φ = radial basis, n_rbf wide (optional)
γ = 1 + Δ,   (β)                                   last layer zero-init ⇒ γ = 1, β = 0 at init

A_cos ← γ ⊙ A_cos + β·δ_{m,0}
A_sin ← γ ⊙ A_sin
```

Equivariance conditions, both enforced in `ecenet/film.py`:

- **Scale**: `cos` and `sin` of the same `(c, m)` must share `γ`. Given that,
  a per-`(c, m)` scale (`film_per_m=True`) is as equivariant as a per-channel
  one, since a scalar commutes with the `e^{imφ}` rotation of that pair.
  Structural-zero slots are masked to `γ = 1` so the gate cannot act on them.
- **Shift**: `β` is added to the `m = 0` slot of `A_cos` **only**. `m = 0` is
  the invariant mode (`e^{i·0·φ} = 1`), so an invariant scalar added there
  is exact; a shift on any `m > 0` mode would break equivariance outright.

### Charge / spin gate (`charge_spin=True`)

Without it the model never sees the electronic state: `E` is a function of
`{r_i}` and `{Z_i}` alone, so a cation and its neutral parent at one geometry
are one input. `charge_spin=True` supplies the structure's total charge `Q` (e)
and total spin `S` (unpaired electrons, `= multiplicity - 1`) as an invariant
vector, per structure `b`:

```
q_b = [ Q_b , S_b , Q_b/N_b , S_b/N_b ]  ∈ ℝ⁴          electronic.state_features
```

`N_b` is the atom count. The intensive half is what a **size-consistent**
conditioning of a sum-of-local-terms energy needs — replicating a system
together with its charge leaves `Q/N` fixed — while the extensive half is
needed because the response to adding one electron is not intensive (§ the
`Q, S` note in `ecenet/electronic.py`).

It enters at two places, both **identity at init**:

**1. A second FiLM gate** (`charge_spin_film`), the same generator as the
element gate with `q_b` replacing the radial leg, run immediately after it:

```
g_e = MLP( [ a(Z_i) ; b(Z_j) ; q_{b(e)} ] )
A_cos ← γ ⊙ A_cos + β·δ_{m,0},   A_sin ← γ ⊙ A_sin
```

Equivariance is the element gate's argument verbatim: `q_b` is an invariant
scalar vector, so `γ` and `β` are invariant scalars, `γ` is shared by `cos` and
`sin` of a given `(c, m)`, and `β` touches `m = 0` only. Because the gate sits
in front of the whole stack, everything downstream — `ε_e`, `l0`/`l1`, the
mixture invariants — becomes state-dependent.

**2. A state-conditioned atomic energy** (`charge_spin_atomic`), added beside
the per-type baseline of §10:

```
a_state(i) = MLP( [ embed(Z_i) ; q_{b(i)} ] )         electronic.StateAtomicEnergy
E ← E + Σ_i a_state(i)
```

Edge-free by construction, so it is the term that survives when every neighbour
has left `r_cut_edge`: an isolated ion stays distinguishable from an isolated
neutral atom, and the energy does not step as the last edge leaves.

Both heads are zero-init at their last layer (`γ = 1`, `β = 0`, `a_state = 0`),
so switching `charge_spin` on leaves a model's step-0 predictions — and, since
the heads are built last, its trunk weights under a fixed seed — exactly as they
were. Omitting `total_charge`/`total_spin` at call time means `Q = S = 0`.

---

## 8. Equivariant layer

One `ECENetLayer` is a residual block: linear → nonlinearity (→ up-projection).

### `EquivariantLinear`

Channel mixing that is **block-diagonal across `m`** — modes never mix, and the
same weight acts on the cos and sin parts:

```
A_cos'[…, o, m] = Σ_c W[m, o, c] · A_cos[…, c, m]  (+ b_o if m = 0)
A_sin'[…, o, m] = Σ_c W[m, o, c] · A_sin[…, c, m]
```

`W` has shape `(n_angular, out, in)`, init `𝒩(0, 2/(in+out))`. The bias lands on
`m = 0` of the cos part only — the same invariant-mode argument as `β` in §7.

### `RealSpaceNonlinearity`

A pointwise nonlinearity applied equivariantly, by going to the angular
coordinate and back (iDFT → σ → DFT) on a uniform grid `θ_k = 2πk/G`:

```
synthesis:  f(θ_k) = Σ_{m=0}^{m_max} A_cos[m]·cos(m θ_k) + A_sin[m]·sin(m θ_k)
apply:      g(θ_k) = σ(f(θ_k))                                σ ∈ {silu, tanh, relu, gelu}
analysis:   A_cos'[m] = ν_m Σ_k g(θ_k)·cos(m θ_k)             ν_0 = 1/G, ν_{m>0} = 2/G
            A_sin'[m] = ν_m Σ_k g(θ_k)·sin(m θ_k)
```

This is equivariant because a rotation of the bond frame by `φ` is exactly the
shift `θ → θ - φ`, and a **pointwise** map commutes with a shift.

`G = 4·m_max + 1` by default — deliberate oversampling, since `σ` generates
harmonics above `m_max` and the analysis step truncates them. The DFT constants
are always computed in `float64` and cast down (rebuilt on every dtype/device
change and after `load_state_dict`), because the equivariance of the layer rests
on the synthesis→analysis round-trip being faithful; building them at the
default `float32` once capped rotational consistency at ~1e-7 regardless of the
working precision.

### The block

```
full width:   A ← A + nonlin(linear(A))
bottleneck:   A ← A + up(nonlin(down(A)))        up zero-init ⇒ identity at init
```

With `bottleneck_dim = r`, the nonlinearity runs at width `r` instead of `F`,
and the block is a low-rank update. `n_layers` such blocks make one **stage**.

---

## 9. Message passing (`n_mp ≥ 2`)

The layer stack alone is per-edge. `n_mp` stages of `n_layers` blocks are
separated by `n_mp - 1` message-passing layers (none after the last stage) that
exchange information between edges through the atoms.

`ECENetAttentionMPLayer` computes, per edge, a **message** and one or more
invariant **scores** from a single fused trunk:

```
u = up( nonlin( down(A) ) )                     up emits n_ch + n_scores channels
m_e = A_e + u[:n_ch]                            message (residual, low-rank update)
s_e = u[n_ch:, m=0]                             scores (m=0 ⇒ invariant)
```

`up` is zero-init, so at initialisation every message residual and every score
is exactly 0.

### Weights

With `f_cut = f_cut(r_ij, r_cut_edge)`:

| `mp_type` | weight `a_e` | character |
| --- | --- | --- |
| `'softmax'` | `exp(s_e)·f_cut / (Σ_{e'→j} exp(s_e')·f_cut_{e'} + ε)` | normalised average — **intensive** in coordination |
| `'sum'` | `s_e · f_cut` | signed weighted sum — **extensive**, and a neighbour can contribute negatively |

The envelope enters multiplicatively either way, so a departing edge's weight
vanishes continuously at `r_cut_edge` (it leaves numerator and normaliser
together). The `+ε` floor (`1e-6`) keeps a node's aggregate finite as its last
edge leaves. Max-subtraction inside the softmax is detached, so the result is an
exact softmax.

`mp_msg_envelope` (default on) multiplies `f_cut` back in for `'softmax'`: the
normaliser divides the *absolute* envelope out, leaving only the relative
envelope across a receiver's in-edges, so without it a lone neighbour near
`r_cut` still gets weight ≈ 1. It is a no-op for `'sum'`, which is enveloped by
construction.

At init: `'sum'` is an exact identity (`s = 0 ⇒ a = 0`); `'softmax'` starts with
uniform attention (`exp(0) = 1`).

### Aggregation

Messages are packed back to spherical slots, **unrotated to the common global
frame**, weighted, and scattered to the receiver atom:

```
h_e = D(r̂_e)ᵀ · pack(m_e)                       bond frame → global frame
Δ_j = Σ_{e : j(e) = j}  a_e ⊙ h_e                 scatter-add over in-edges
```

then gathered at each edge's **source** atom, rotated back into that edge's bond
frame, and passed through a residual receiver block:

```
d_e  = unpack( Δ_{i(e)} · D(r̂_e) )
A_e ← A_e + receiver(d_e)                        receiver = bottleneck ECENetLayer
```

The cross-edge sum happens in the global frame, which is what makes it legal —
each edge's features are expressed in a *different* bond frame and could not be
added directly.

### Heads and per-`l` attention

With `n_heads = H`, the score head emits `H` scores per edge and the value
channels `n_base` split into `H` contiguous groups of whole spherical channels
(`n_base` must be divisible by `H`). Head `h` weights the receiver's in-edges
independently and gates only its own slice.

With `mp_l_attention=True`, each head emits one score **per degree `l`**, and
each `(head, l)` runs its own softmax; the trunk widens to `n_ch +
H·(l_max+1)`. The score is still an invariant scalar applied uniformly across
that `l`'s whole `m`-block, expanded through a fixed `l_of_s` map. This is
equivariant precisely because `D_block` is `l`-diagonal — splitting across `l`
is legal, splitting *within* an `l` across `m` is not.

---

## 10. Invariants and read-out

The `m = 0` component is rotation-invariant. The read-out takes it directly:

```
h_e = A_cos[e, :, 0]  ∈ ℝ^F
```

Per-edge energy, with the same radial recipe as the basis so the contribution
vanishes smoothly at `r_cut_edge`:

```
n_max_d set:     ε_e = Σ_d MLP(h_e)_d · f_d(r_ij)         learned distance profile
n_max_d = None:  ε_e = MLP(h_e) · f_cut(r_ij)             scalar × envelope
```

Total energy:

```
E = Σ_{e ∈ E} ε_e  +  Σ_i  a[Z_i]  ( +  Σ_i a_state(i)  with charge_spin )
```

Edges are directed and both orientations are present, so every pair contributes
twice — the read-out is a sum over *directed* edges, not over pairs. `a` is the
per-type atomic-energy baseline (`atomic_energy`, zero-init). `OutputMLP`'s last
layer is near-zero-init (`std = 0.01`), so per-edge energies start near zero and
the baseline dominates early training.

**Zero-edge structures** keep `Σ_i a[Z_i]` (and, under a mixture read-out, the
per-`(type, expert)` constants). Returning a bare 0 would make the energy jump
by `Σ a[Z_i]` as the last edge crosses `r_cut_edge`.

---

## 11. Mixture-of-experts read-out (`n_experts = K > 1`)

`MixtureReadout` (`ecenet/moe.py`) replaces the single head with `K` **diabatic**
expert heads over the same shared invariants, plus a mixing rule. `K = 1` is a
strict no-op: no head is built, no parameters exist, numerics are unchanged.

A **segment** is the unit that gets its own `K × K` problem: an atom
(`scope='atom'`, default) or a whole structure (`scope='global'`).

### Expert energies and couplings

Both are the §10 read-out with a wider output block, summed over the segment,
plus a per-type constant:

```
V_a[σ] = Σ_{i ∈ σ} v[Z_i, a]  +  Σ_{e ∈ σ} ε_a(h_e, r_e)          a = 1..K
C_p[σ] = Σ_{i ∈ σ} c[Z_i, p]  +  Σ_{e ∈ σ} γ_p(h_e, r_e)          p = 1..P
```

`P` is set by the coupling topology: `full` → `K(K-1)/2`, `chain` → `K-1`
(tridiagonal `H`), `none` → `0`.

### Mixing rules

```
H_ab = δ_ab V_a + (1 - δ_ab) C_ab                real symmetric, K × K

'evb'      E = λ_min(H),               w = c₀²            coupled variational selection
'moe'      E = Σ_a w_a V_a,            w = softmax(gate)  ordinary MoE
'softmin'  E = -τ log Σ_a exp(-V_a/τ), w = softmax(-V/τ)  entropic smoothing of min
'mean'     E = (1/K) Σ_a V_a,          w = 1/K            no gating
```

All four share identical expert heads, so an ablation changes one string.

`softmin`'s temperature is **extensive**: `τ = moe_tau · max(|σ|, 1)` with `|σ|`
the segment's atom count. An extensive energy needs an extensive temperature, or
the smoothing vanishes as `N` grows and softmin silently degenerates into a hard
`min`.

### The `K = 2` closed form

```
E₀ = ½(V_A + V_B) - √( ¼(V_A - V_B)² + C² + ε_gap )
w_A = ½(1 - d/r),   d = ½(V_A - V_B),   r = √(d² + C² + ε_gap)
```

A hyperbolic regularisation of `min(V_A, V_B)`: far from a crossing the lower
expert dominates; near one the coupling opens an avoided crossing of width
`2|C|`. Unlike a convex mixture, the coupled energy lies **below** every expert
(variational). Used in preference to the eigensolver at `K = 2` because it is
analytic — the second derivatives that force training needs come out of ordinary
autograd with no eigenvector-perturbation term. `K > 2` goes through
`torch.linalg.eigh`.

### Hellmann–Feynman

Since `E` is a plain scalar function of the positions, autograd forces are
exactly `-∇E`, and

```
F = -c₀ᵀ (∇H) c₀ = -Σ_a w_a ∇V_a - Σ_{a≠b} c₀a c₀b ∇C_ab
```

so no separate force-gating network is needed and energy conservation is
structural. The **eigenvalue** gradient is `c₀c₀ᵀ` and needs no gap; the
**second** derivative differentiates the eigenvector and scales as `1/(E₁-E₀)`.
Nonzero couplings (`moe_coupling_init > 0`) keep that gap open — the degenerate
case to watch for is couplings driven to zero with two experts crossing.

### Size consistency

For two non-interacting subsystems the global-scope `V` and `C` are sums over
atoms, so `H_total = H_A + H_B` — and `λ_min` is superadditive under matrix
addition (Weyl; immediately from the Rayleigh quotient):

```
λ_min(H_A + H_B) ≥ λ_min(H_A) + λ_min(H_B)
```

So `scope='global'` is **not size-consistent** — the pair does not give the sum
of the separated energies, and the mixing weights of a large cell
are set by whole-cell energy differences that grow with `N`. `scope='atom'` is
exactly additive and keeps the couplings intensive; it is the default.
`tests/test_moe.py` asserts both behaviours.

### Gauge freedom

`H + f(R)·I` shifts every eigenvalue by `f(R)`. The shared `atomic_energy`
baseline therefore sits **outside** the Hamiltonian, exactly separable from the
mixture.

### Expert collapse

Nothing prevents one expert from sitting below all others everywhere, at which
point `c₀ → (1, 0, …, 0)`. `diversity_loss(weights, kind)` supplies the
counter-pressure a trainer adds to the data loss:

```
'load'     K · Σ_a f_a p_a            Switch-Transformer load balancing; f = win
                                      fraction, p = mean weight. Pushes on the batch
                                      marginal, not per-segment sharpness (= 1 when uniform)
'entropy'  H̄_seg - H(p̄)               sharp per-configuration, spread across the batch
                                      — i.e. specialisation
'cv'       std(p̄)/mean(p̄)             softest; blind to per-assignment sharpness
```

---

## 12. Long-range electrostatics (optional)

With the LES add-on the total is evaluated on **one** autograd graph:

```
E = E_sr + E_lr
```

`E_sr` is §10/§11. `E_lr` is the smeared-Coulomb energy of latent charges (and,
optionally, latent atomic dipoles) predicted from the per-atom invariant
embedding `l0`, reciprocal-space Ewald under PBC. Because both terms share the
graph, forces and stress need no extra code.

### Aggregating `l0`

The per-atom embedding is a scatter-sum of edge invariants at the receiver.
`les_readout` selects the weight:

| mode | `l0` | note |
| --- | --- | --- |
| `'sum'` | `Σ_{e→j} h_e` | parameter-free, extensive |
| `'softmax'` | `Σ_{e→j} a_e h_e` | attention weights mirroring §9, `a_e = exp(s)f_cut/(Σ exp(s)f_cut + ε)·f_cut` |
| `'edge'` | `Σ_{e→j} q_e`, width 1 | linear head on `h_e`; `l0` **is** the charge |
| `'edge_basis'` | `Σ_{e→j} q_e`, width 1 | MLP head mirroring §10 end to end: `n_max_d` channels dotted with the enveloped radial basis, so each bond's contribution has a learnable distance profile and vanishes exactly at `r_cut` |

The `l1` (vector) embedding always uses the plain unweighted sum, obtained by
applying `D^1ᵀ` to the `l=1` block only — much cheaper than the full `D^{l_max}`.

Two initialisation notes, both consequences of `E_lr` being **quadratic** in the
charges:

- The charge head is **not** zero-init. `q ≡ 0` is a gradient-free saddle a
  zero-init head could never leave.
- The dipole block **is** zero-init when charges are on, because the `qᵀf_qu·u`
  cross-term drives it once charges exist. In the dipoles-only ablation
  (`les_charges=False`) that cross-term is gone (`∂E_lr/∂u ∝ u`), so the dipole
  block reverts to standard init.

`les_charge_scale` multiplies the edge-mode latent charge by a fixed `s`, which
suppresses `E_lr` by ~`s²` early so the short-range fit leads.

---

## 13. Derived quantities

### Forces

```
F_i = -∂E/∂r_i
```

by autograd through the whole graph. Training on forces differentiates the model
twice, which is why: the ACE backward is analytic (§3), `'poly'` cutoffs are C²
(§2), the `K=2` EVB eigenvalue is closed-form (§11), and the fused activation
path (`set_activation_fused`) must stay **off** — it is single-backward only.
`set_edge_frame_fused` is safe for double-backward: its backward is composed of
differentiable ops.

### Stress

Symbolic infinitesimal strain, exact and linear at `ε = 0`:

```
r → r + r·ε,      S_e → S_e + S_e·ε,      h → h + h·ε   (cell, LES only)
σ = (1/V) · ∂E/∂ε |_{ε=0}
```

One backward pass yields both `-∂E/∂r` and `∂E/∂ε`. The neighbour topology is
frozen across the strain (the standard MLIP approximation). Reported to ASE in
Voigt order `(xx, yy, zz, yz, xz, xy)`.

---

## 14. Why each step is equivariant

Reading the pipeline as a chain of representations:

| Stage | Symmetry used | Why it holds |
| --- | --- | --- |
| ACE basis (§3) | SO(3) | `Y_lm` transform by `D^l`; the sum over neighbours is linear |
| Embedding (§4) | SO(3) | contraction over `(t, n)` only — never over `s` |
| Bond frame (§5) | SO(3) → SO(2) | `D(r̂)` is built *from* the geometry, so it co-rotates |
| Angular split (§6) | SO(2) | mode `m` carries `e^{imφ}` |
| `EquivariantLinear` (§8) | SO(2) | block-diagonal in `m`; shared cos/sin weights; bias on `m=0` only |
| `RealSpaceNonlinearity` (§8) | SO(2) | rotation = shift `θ → θ - φ`; pointwise maps commute with shifts |
| FiLM gates (§7) | SO(2) | scale shared by the cos/sin pair; shift on `m=0` only |
| Charge/spin (§7) | — | `[Q, S, Q/N, S/N]` is invariant, so it only ever feeds invariant `γ`, `β` and atomic scalars |
| MP weights (§9) | SO(3) | scores are `m=0`; `f_cut` depends on an invariant distance |
| MP aggregation (§9) | SO(3) | cross-edge sum performed in the common global frame |
| Per-`l` attention (§9) | SO(3) | `D_block` is `l`-diagonal; weight uniform across each `m`-block |
| Read-out (§10) | — | `m = 0` is invariant, hence so is `E` |

Forces and stress are then equivariant automatically, as gradients of an
invariant scalar.

---

## 15. Shapes and parameter counts

### Tensor shapes through one forward

| Quantity | Shape |
| --- | --- |
| `A` (ACE basis) | `(N, n_types, n_max, n_sph)` |
| `A_emb` | `(N, embed_dim, n_sph)` |
| `A_rot` | `(E, n_base, n_sph)` |
| `A_cos`, `A_sin` | `(E, F, n_angular)` |
| `D_block` | `(E, n_sph, n_sph)` |
| `h_e` (invariants) | `(E, F)` |
| `l0` / `l1` | `(N, 2·embed_dim)` or `(N, 1|4)` / `(N, 2·embed_dim, 3)` |
| `V` / `C` (mixture) | `(n_seg, K)` / `(n_seg, P)` |
| `H` (mixture) | `(n_seg, K, K)` |

### Parameters

| Module | Count |
| --- | --- |
| `W` | `n_types² · n_max · embed_dim` |
| `EquivariantLinear(in, out)` | `n_angular · out · in + out` |
| `ECENetLayer` (full) | `n_angular · F² + F` |
| `ECENetLayer` (bottleneck `r`) | `2 · n_angular · F · r + r + F` |
| `RealSpaceNonlinearity` | 0 (buffers only) |
| MP layer | receiver (bottleneck `mp_dim`) + `msg_down` + `msg_up(mp_dim → F + n_scores)` |
| `OutputMLP([F, …, n_out])` | standard dense MLP |
| `atomic_energy` | `n_types` |
| `MixtureReadout` | `expert_net(→ K·n_out)` + `K·n_types` + `coupling_net(→ P·n_out)` + `P·n_types` (+ gate) |

`n_scores = n_heads` (or `n_heads·(l_max+1)` with `mp_l_attention`).

The mixture read-out replaces `output_net` rather than adding to it: at
`n_experts > 1` the single head is not built at all, so parameter counts stay
comparable.

---

## 16. Symbol → code map

| Symbol | Code |
| --- | --- |
| `f_cut` | `radial.get_cutoff_fn` |
| `f_n(r)` | `radial.radial_basis` |
| `A[i,t,n,s]` | `ace_basis.ACEBasisAnalytic` |
| `W` | `ECENet.W`, applied in `ECENet._embed` |
| `D^1` | `spherical.build_D1_from_rhat` |
| `D^l` | `spherical.recursive_wigner_D` |
| `D_block` | `spherical.build_D_block` |
| `A_rot` | `spherical.wigner_rotate`, or `ECENet._edge_frame` |
| `A_cos`, `A_sin` | `model.SphToAngular` |
| `γ`, `β` | `film.ElementFiLM` |
| `q_b`, `a_state` | `electronic.state_features`, `electronic.StateAtomicEnergy` |
| linear / nonlinearity | `equivariant.EquivariantLinear`, `equivariant.RealSpaceNonlinearity` |
| residual block | `model.ECENetLayer` |
| message passing | `model.ECENetAttentionMPLayer` |
| `h_e` | `ECENet._contract` |
| `ε_e` | `ECENet._apply_output` |
| `E` | `ECENet._readout_energy` |
| `l0`, `l1` | `ECENet._aggregate_lr_embeddings` |
| `H`, `λ_min(H)` | `moe.build_hamiltonian`, `moe.evb_ground_state` |
| `K = 2` closed form | `moe.evb_two_state` |
| collapse regularisers | `moe.diversity_loss` |
| `σ` (stress) | `calculator.ECENetCalculator._compute_stress` |

---

## 17. Hyperparameter defaults

| Name | Default | Effect |
| --- | --- | --- |
| `r_cut_edge` | 5.0 Å | which pairs become edges |
| `r_cut_neighbor` | 4.0 Å | which atoms enter the ACE basis |
| `l_max` | 3 | degree of the SH / ACE basis |
| `m_max` | `l_max` | angular modes kept after §6 |
| `n_max` | 4 | radial functions per `(type, l)` |
| `embed_dim` | 16 | width after the `(n_types, n_max)` contraction |
| `n_layers` | 2 | equivariant blocks per stage |
| `n_mp` | 1 | stages; `≥ 2` turns on message passing |
| `n_max_d` | `None` | read-out radial rank |
| `cutoff_type` | `'cosine'` | C¹; use `'poly'` (C²) for force training |
| `activation` | `'silu'` | pointwise nonlinearity |
| `mp_type` | `'softmax'` | intensive vs. extensive aggregation |
| `mp_dim` | `max(F // 4, 1)` | MP trunk / receiver bottleneck |
| `mp_n_heads` | 1 | attention heads (must divide `2·embed_dim`) |
| `output_hidden_dims` | `[64]` | read-out MLP widths |
| `charge_spin` | `False` | `True` conditions on total charge / spin |
| `n_experts` | 1 | `> 1` switches to the mixture read-out |
| `moe_mixture` | `'evb'` | mixing rule |
| `moe_scope` | `'atom'` | size-consistent; `'global'` is not |
| `moe_coupling_init` | 0.05 eV | opens the `E₁-E₀` gap at init |
