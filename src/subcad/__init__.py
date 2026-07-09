from .detection import GreedyDetector, GreedyPPDetector, SpectralDetector
from .selection import DensitySelector, SpectralSeededSelector
from .aggregators import MajorityVoting, DawidSkene, WeightedMajorityVoting, WeightedDawidSkene

from pathlib import Path
PROJECT_DIR = str(Path(__file__).parents[2])
