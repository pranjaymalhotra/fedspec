"""
Tests package initialization.
"""
from .test_svd_optimality import TestSVDOptimality
from .test_aggregation_bias import TestAggregationBias

__all__ = [
    "TestSVDOptimality",
    "TestAggregationBias"
]
