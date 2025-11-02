from ._condensation import MCNN
from ._edition import AllKNN
from ._hybrid import ICF

__all__ = ["Reductor"]

class Reductor:
    MCNN = MCNN
    ALLKNN = AllKNN
    ICF = ICF
