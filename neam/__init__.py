"""Neam (នាម): Khmer news named entity recognition."""
from neam.inference import DEFAULT_MODEL, KhmerNER, Neam
from neam.preprocess import LABELS, segment_document

__all__ = ['DEFAULT_MODEL', 'KhmerNER', 'LABELS', 'Neam', 'segment_document']
__version__ = '0.1.0'
