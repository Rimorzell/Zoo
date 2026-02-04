"""Data models for match results."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
from datetime import datetime


class ConfidenceLevel(Enum):
    """Classification of match confidence for human review routing."""
    AUTO_APPROVE = "AUTO_APPROVE"
    REVIEW_RECOMMENDED = "REVIEW_RECOMMENDED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    NO_MATCH = "NO_MATCH"


@dataclass
class ConstraintCheck:
    """Result of a single constraint check."""
    constraint: str
    status: str  # PASS, FAIL, N/A
    explanation: str

    def to_dict(self) -> Dict:
        return {
            "constraint": self.constraint,
            "status": self.status,
            "explanation": self.explanation,
        }


@dataclass
class Alternative:
    """An alternative product match."""
    sku: str
    name: str
    confidence: float
    trade_off: str

    def to_dict(self) -> Dict:
        return {
            "sku": self.sku,
            "name": self.name,
            "confidence": self.confidence,
            "trade_off": self.trade_off,
        }


@dataclass
class MatchResult:
    """Complete result of matching a requirement to a product."""
    line_item: str
    quantity: int
    unit: str

    matched_sku: Optional[str] = None
    matched_product_name: Optional[str] = None

    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.NO_MATCH
    confidence_reason: Optional[str] = None  # "DERIVED_FROM_BASE", "DIRECT_MATCH", etc.

    reasoning: str = ""

    constraint_checks: List[ConstraintCheck] = field(default_factory=list)
    alternatives: List[Alternative] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    needs_human_review: bool = True
    review_reason: Optional[str] = None

    source_confidence: float = 1.0
    matching_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Additional context
    candidates_evaluated: int = 0
    selection_method: str = ""  # "single_candidate", "llm_selection", "no_candidates"

    # Inheritance tracking
    is_derived_match: bool = False
    base_item: Optional[str] = None
    base_sku: Optional[str] = None
    inherited_constraints: List[str] = field(default_factory=list)
    overridden_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        result = {
            "line_item": self.line_item,
            "quantity": self.quantity,
            "unit": self.unit,
            "matched_sku": self.matched_sku,
            "matched_product_name": self.matched_product_name,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "confidence_reason": self.confidence_reason,
            "reasoning": self.reasoning,
            "constraint_checks": {
                check.constraint: f"{check.status} - {check.explanation}"
                for check in self.constraint_checks
            },
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "warnings": self.warnings,
            "needs_human_review": self.needs_human_review,
            "review_reason": self.review_reason,
            "source_confidence": self.source_confidence,
            "matching_timestamp": self.matching_timestamp,
            "candidates_evaluated": self.candidates_evaluated,
            "selection_method": self.selection_method,
        }

        # Add inheritance info if this is a derived match
        if self.is_derived_match:
            result["inheritance"] = {
                "is_derived_match": True,
                "base_item": self.base_item,
                "base_sku": self.base_sku,
                "inherited_constraints": self.inherited_constraints,
                "overridden_constraints": self.overridden_constraints,
            }

        return result


@dataclass
class MatchingSummary:
    """Summary of all matching results."""
    total_requirements: int = 0
    matched: int = 0
    needs_clarification: int = 0
    no_match: int = 0
    review_required: int = 0

    by_confidence_level: Dict[str, int] = field(default_factory=dict)
    action_required: List[Dict] = field(default_factory=list)
    matches: List[MatchResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "processing_summary": {
                "total_requirements": self.total_requirements,
                "matched": self.matched,
                "needs_clarification": self.needs_clarification,
                "no_match": self.no_match,
                "review_required": self.review_required,
            },
            "by_confidence_level": self.by_confidence_level,
            "action_required": self.action_required,
            "matches": [m.to_dict() for m in self.matches],
        }
