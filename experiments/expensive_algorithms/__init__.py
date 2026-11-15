"""Independent algorithm entry points for the expensive suite.

Each module exposes one method family.  The experiment runner imports these
entry points instead of embedding method selection logic in the workbook code.
"""

from .esa import ESAPort
from .llm_operator_de import run_ablation_algorithm
from .bgp_saea import RecentPaperController
from .paper_surrogates import run_paper_algorithm
from .sade_ss import SADE_SSController
from .sgdlco import SGDLCOController
from .tsddeo import TSDDEOPort

__all__ = [
    "TSDDEOPort",
    "ESAPort",
    "SADE_SSController",
    "SGDLCOController",
    "RecentPaperController",
    "run_paper_algorithm",
    "run_ablation_algorithm",
]
