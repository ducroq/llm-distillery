"""
Commerce Prefilter v2 - Embedding + MLP Classifier

Uses frozen sentence-transformers embeddings with a trained MLP classifier.
Achieves same accuracy as v1 (97.8% F1) with simpler architecture.
"""

from .inference import CommercePrefilterV2, is_commerce

__all__ = ['CommercePrefilterV2', 'is_commerce']
__version__ = '2.0.0'
