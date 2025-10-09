from ._base_metrics import BaseMetrics
from ._heterogeneous_metrics import IVDM, HEOM, GWHSM

__all__ = ["Metrics"]

class Metrics:
    # Unified metrics interface
    Base = BaseMetrics
    IVDM = IVDM
    HEOM = HEOM
    GWHSM = GWHSM