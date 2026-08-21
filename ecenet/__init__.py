"""ECENet — an SO(3)-equivariant interatomic potential (MLIP) using per-edge
SO(2) features.

Public API:
    ECENet               — the model (message passing on when n_mp >= 2)
    MultiECENet          — several ECENet diabats mixed by an empirical valence
                           bond matrix; ground state = lowest eigenvalue
    ECENetCalculator     — ASE calculator wrapper (lazy import; needs `ase`)
    MultiECENetCalculator — ASE calculator for EVB checkpoints; reads the
                           sector from atoms.info['charge'] / ['spin']
    ECENetLESCalculator  — ASE calculator for joint-LES checkpoints
                           (E_sr + E_lr; lazy; needs `ase` and `les`)
    LESLongRange         — optional LES long-range add-on (lazy; needs `les`)
"""

from ecenet.evbmodel import MultiECENet
from ecenet.model import ECENet

__all__ = ["ECENet", "MultiECENet", "ECENetCalculator",
           "MultiECENetCalculator", "ECENetLESCalculator", "LESLongRange"]


def __getattr__(name):
    # Lazy so `import ecenet` doesn't require ASE (calculator) or the optional
    # `les` package (LES add-on) unless they are used.
    if name == "ECENetCalculator":
        from ecenet.calculator import ECENetCalculator
        return ECENetCalculator
    if name == "MultiECENetCalculator":
        from ecenet.calculator import MultiECENetCalculator
        return MultiECENetCalculator
    if name == "ECENetLESCalculator":
        from ecenet.calculator import ECENetLESCalculator
        return ECENetLESCalculator
    if name == "LESLongRange":
        from ecenet.les import LESLongRange
        return LESLongRange
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
