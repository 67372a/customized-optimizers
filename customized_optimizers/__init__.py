
from typing import Dict, List
from .utils import OPTIMIZER
from .simplifiedademamix import (SimplifiedAdEMAMix, SimplifiedAdEMAMixExM)
from .came import CAME
from .fftdescent import FFTDescent
from .singstate import SingState
from .talon import TALON

OPTIMIZER_LIST: List[OPTIMIZER] = [
    CAME,
    FFTDescent,
    SimplifiedAdEMAMix,
    SimplifiedAdEMAMixExM,
    SingState,
    TALON
]

OPTIMIZERS: Dict[str, OPTIMIZER] = {str(f"{optimizer.__name__}".lower()): optimizer for optimizer in OPTIMIZER_LIST}