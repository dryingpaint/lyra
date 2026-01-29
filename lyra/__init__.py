"""
Lyra: Efficient Subquadratic Architecture for Biological Sequence Modeling

A hybrid architecture combining Projected Gated Convolutions (PGC) for local 
feature extraction with State Space Models (S4D) for capturing long-range 
dependencies. Designed for modeling epistatic interactions in biological sequences.

Paper: "Lyra: An Efficient and Expressive Subquadratic Architecture for 
Modeling Biological Sequences" - Ramesh, Siddiqui, Gu, Mitzenmacher, Sabeti
"""

from .model import Lyra
from .s4d import S4D, S4DKernel
from .pgc import PGC

__version__ = "0.1.0"
__all__ = ["Lyra", "S4D", "S4DKernel", "PGC"]
