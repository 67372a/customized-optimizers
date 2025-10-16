
from typing import Dict, List
from .utils import OPTIMIZER
from .simplifiedademamix import (SimplifiedAdEMAMix, SimplifiedAdEMAMixExM)
from .came import CAME
from .fftdescent import FFTDescent
from .singstate import SingState
from .talon import TALON
from .scgopt import SCGOpt
from .ocgopt import OCGOpt

OPTIMIZER_LIST: List[OPTIMIZER] = [
    CAME,
    FFTDescent,
    OCGOpt,
    SCGOpt,
    SimplifiedAdEMAMix,
    SimplifiedAdEMAMixExM,
    SingState,
    TALON
]

OPTIMIZERS: Dict[str, OPTIMIZER] = {str(f"{optimizer.__name__}".lower()): optimizer for optimizer in OPTIMIZER_LIST}