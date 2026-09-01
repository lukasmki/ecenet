"""ECENet — an SO(3)-equivariant interatomic potential (MLIP) using per-edge
SO(2) features.

Public API:
    ECENet               — the model (message passing on when n_mp >= 2)
    MixtureReadout       — K-expert read-out (EVB / MoE / softmin); ECENet(n_experts>1)
    diversity_loss       — expert-collapse regulariser for mixture training
    ECENetCalculator     — ASE calculator wrapper (lazy import; needs `ase`)
    ECENetLESCalculator  — ASE calculator for joint-LES checkpoints
                           (E_sr + E_lr; lazy; needs `ase` and `les`)
    LESLongRange         — optional LES long-range add-on (lazy; needs `les`)
"""

from ecenet.model import ECENet
from ecenet.moe import MixtureReadout, diversity_loss

__all__ = ["ECENet", "MixtureReadout", "diversity_loss",
           "ECENetCalculator", "ECENetLESCalculator", "LESLongRange"]


def __getattr__(name):
    # Lazy so `import ecenet` doesn't require ASE (calculator) or the optional
    # `les` package (LES add-on) unless they are used.
    if name == "ECENetCalculator":
        from ecenet.calculator import ECENetCalculator
        return ECENetCalculator
    if name == "ECENetLESCalculator":
        from ecenet.calculator import ECENetLESCalculator
        return ECENetLESCalculator
    if name == "LESLongRange":
        from ecenet.les import LESLongRange
        return LESLongRange
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
