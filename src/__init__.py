"""
Speedy Gram2Vec - A fast implementation of grammatical style embedding

This module provides the same interface as the original gram2vec package,
implementing document embedding based on grammatical style features.
"""

from .vectorizer import Gram2VecVectorizer, default_config

# Export main classes
__all__ = ['Gram2VecVectorizer', 'default_config']

# For backward compatibility, also export as vectorizer
vectorizer = Gram2VecVectorizer
