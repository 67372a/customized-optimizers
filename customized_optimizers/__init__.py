
from typing import Dict, List
from .utils import OPTIMIZER
from .simplifiedademamix import (SimplifiedAdEMAMix, SimplifiedAdEMAMixExM)
from .came import CAME
from .fftdescent import FFTDescent
from .singstate import SingState
from .talon import TALON
from .scgopt import SCGOpt
from .ocgopt import OCGOpt
from .oagopt import OAGOpt
from .snoo_asgd import SNOO_ASGD
from .abmog import ABMOG

OPTIMIZER_LIST: List[OPTIMIZER] = [
    ABMOG,
    CAME,
    FFTDescent,
    OAGOpt,
    OCGOpt,
    SCGOpt,
    SimplifiedAdEMAMix,
    SimplifiedAdEMAMixExM,
    SingState,
    SNOO_ASGD,
    TALON
]

OPTIMIZERS: Dict[str, OPTIMIZER] = {str(f"{optimizer.__name__}".lower()): optimizer for optimizer in OPTIMIZER_LIST}