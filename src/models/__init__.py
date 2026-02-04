"""Data models for the lighting product matching system."""

from .requirement import Requirement, RequirementSpecs, WattageSpec, LumensSpec, ColorTempSpec
from .product import Product
from .match_result import MatchResult, ConstraintCheck, Alternative, ConfidenceLevel

__all__ = [
    "Requirement",
    "RequirementSpecs",
    "WattageSpec",
    "LumensSpec",
    "ColorTempSpec",
    "Product",
    "MatchResult",
    "ConstraintCheck",
    "Alternative",
    "ConfidenceLevel",
]
