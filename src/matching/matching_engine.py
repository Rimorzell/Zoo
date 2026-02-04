"""Main matching engine orchestrator.

Coordinates all components to match requirements to products.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any

from ..models.requirement import Requirement
from ..models.product import Product
from ..models.match_result import (
    MatchResult, MatchingSummary, ConstraintCheck,
    Alternative, ConfidenceLevel
)
from ..validation.source_validator import SourceValidator, ValidationResult
from .constraint_filter import ConstraintFilter
from .preference_scorer import PreferenceScorer, ScoredCandidate
from .llm_selector import LLMSelector
from .confidence_classifier import ConfidenceClassifier


@dataclass
class MatchingConfig:
    """Configuration for the matching engine."""
    # Validation thresholds
    skip_validation: bool = False
    min_source_confidence: float = 0.5

    # Filtering options
    strict_dimensions: bool = True
    allow_ip_tolerance: bool = True

    # Selection options
    use_llm_selection: bool = True
    llm_backend: Optional[Any] = None

    # Output options
    include_all_candidates: bool = False
    max_alternatives: int = 3


class MatchingEngine:
    """
    Main engine for matching lighting requirements to products.

    Orchestrates the full matching pipeline:
    1. Source validation
    2. Hard constraint filtering
    3. Soft preference scoring
    4. LLM/rule-based selection
    5. Confidence classification
    6. Result generation
    """

    def __init__(self, config: Optional[MatchingConfig] = None):
        """Initialize the matching engine with optional configuration."""
        self.config = config or MatchingConfig()

        # Initialize components
        self.validator = SourceValidator()
        self.filter = ConstraintFilter()
        self.scorer = PreferenceScorer()
        self.selector = LLMSelector(llm_backend=self.config.llm_backend)
        self.classifier = ConfidenceClassifier()

        # Product catalog
        self.products: List[Product] = []

    def load_catalog(self, catalog_path: str) -> int:
        """
        Load product catalog from JSON file.

        Args:
            catalog_path: Path to catalog JSON file

        Returns:
            Number of products loaded
        """
        with open(catalog_path, 'r') as f:
            data = json.load(f)

        products_data = data.get("products", [])
        self.products = [Product.from_dict(p) for p in products_data]

        return len(self.products)

    def load_catalog_from_dict(self, catalog_data: Dict) -> int:
        """
        Load product catalog from dictionary.

        Args:
            catalog_data: Catalog data dictionary

        Returns:
            Number of products loaded
        """
        products_data = catalog_data.get("products", [])
        self.products = [Product.from_dict(p) for p in products_data]

        return len(self.products)

    def match_single(
        self,
        requirement: Requirement,
        validation_result: Optional[ValidationResult] = None
    ) -> MatchResult:
        """
        Match a single requirement to the best product.

        Args:
            requirement: The requirement to match
            validation_result: Pre-computed validation result (optional)

        Returns:
            MatchResult with match details
        """
        # Initialize result
        result = MatchResult(
            line_item=requirement.line_item,
            quantity=requirement.quantity,
            unit=requirement.unit,
            source_confidence=requirement.confidence,
        )

        # Step 1: Validate source if not provided
        if validation_result is None and not self.config.skip_validation:
            validation_result = self.validator.validate(requirement)

        # Check if we can proceed with matching
        if validation_result and not validation_result.can_match:
            result.confidence_level = ConfidenceLevel.NEEDS_CLARIFICATION
            result.reasoning = "Cannot match: " + "; ".join(
                [i.message for i in validation_result.issues
                 if i.severity.value in ["critical", "error"]]
            )
            result.needs_human_review = True
            result.review_reason = "Source data requires clarification"
            result.warnings = [i.message for i in validation_result.issues]
            return result

        # Step 2: Filter products by hard constraints
        filter_result = self.filter.filter(requirement, self.products)
        result.candidates_evaluated = len(filter_result.candidates)

        if not filter_result.candidates:
            result.confidence_level = ConfidenceLevel.NO_MATCH
            result.reasoning = self._build_no_match_reasoning(filter_result)
            result.needs_human_review = True
            result.review_reason = "No products match requirements"
            result.warnings = [
                f"Eliminated {p.name}: {reason}"
                for p, reason in filter_result.eliminated[:5]
            ]
            return result

        # Step 3: Score candidates
        scored_candidates = self.scorer.score_candidates(
            requirement, filter_result.candidates
        )

        # Step 4: Select best match
        if len(scored_candidates) == 1:
            # Single candidate - direct selection
            result.selection_method = "single_candidate"
            best = scored_candidates[0]
            result.matched_sku = best.product.sku
            result.matched_product_name = best.product.name
            result.confidence = best.score
            result.reasoning = f"Single matching product: {best.product.name}"
            result.constraint_checks = filter_result.constraint_checks
            result.warnings = best.notes

        else:
            # Multiple candidates - use selector
            if self.config.use_llm_selection and self.selector.needs_llm_selection(scored_candidates):
                result.selection_method = "llm_selection"
            else:
                result.selection_method = "rule_based_selection"

            selection = self.selector.select(requirement, scored_candidates)

            result.matched_sku = selection.selected_sku
            result.matched_product_name = selection.selected_name
            result.confidence = selection.confidence
            result.reasoning = selection.reasoning
            result.warnings = selection.warnings

            # Add alternatives
            for alt in selection.alternatives[:self.config.max_alternatives]:
                result.alternatives.append(Alternative(
                    sku=alt["sku"],
                    name=alt.get("name", ""),
                    confidence=alt["confidence"],
                    trade_off=alt.get("note", ""),
                ))

        # Step 5: Classify confidence
        result = self.classifier.apply_to_result(result, validation_result)

        # Add constraint checks
        if filter_result.constraint_checks:
            result.constraint_checks = filter_result.constraint_checks[:6]  # Top checks

        return result

    def match_all(
        self,
        requirements: List[Requirement]
    ) -> MatchingSummary:
        """
        Match all requirements to products.

        Args:
            requirements: List of requirements to match

        Returns:
            MatchingSummary with all results and summary statistics
        """
        summary = MatchingSummary()
        summary.total_requirements = len(requirements)

        # Batch validate first
        validation_results = {}
        if not self.config.skip_validation:
            for req in requirements:
                validation_results[req.line_item] = self.validator.validate(req)

        # Match each requirement
        for req in requirements:
            validation = validation_results.get(req.line_item)
            result = self.match_single(req, validation)
            summary.matches.append(result)

            # Update counters
            if result.confidence_level == ConfidenceLevel.NEEDS_CLARIFICATION:
                summary.needs_clarification += 1
            elif result.confidence_level == ConfidenceLevel.NO_MATCH:
                summary.no_match += 1
            elif result.needs_human_review:
                summary.review_required += 1
                summary.matched += 1
            else:
                summary.matched += 1

            # Count by level
            level_key = result.confidence_level.value
            if level_key not in summary.by_confidence_level:
                summary.by_confidence_level[level_key] = 0
            summary.by_confidence_level[level_key] += 1

            # Track action items
            if result.needs_human_review:
                action = {
                    "line_item": result.line_item,
                    "action": self._get_action_type(result),
                    "reason": result.review_reason,
                    "quantity": result.quantity,
                }
                if result.matched_sku:
                    action["matched_sku"] = result.matched_sku
                summary.action_required.append(action)

        return summary

    def _build_no_match_reasoning(self, filter_result) -> str:
        """Build explanation for why no products matched."""
        if not filter_result.eliminated:
            return "No products in catalog"

        # Count elimination reasons
        reasons = {}
        for product, reason in filter_result.eliminated:
            # Extract constraint type from reason
            constraint = reason.split()[0] if reason else "Unknown"
            if constraint not in reasons:
                reasons[constraint] = 0
            reasons[constraint] += 1

        # Build message
        parts = []
        for constraint, count in sorted(reasons.items(), key=lambda x: -x[1]):
            parts.append(f"{count} products failed {constraint}")

        return "No match found. " + "; ".join(parts[:3])

    def _get_action_type(self, result: MatchResult) -> str:
        """Determine action type for review queue."""
        if result.confidence_level == ConfidenceLevel.NEEDS_CLARIFICATION:
            return "CLARIFY_SPECS"
        elif result.confidence_level == ConfidenceLevel.NO_MATCH:
            return "SOURCE_PRODUCT"
        elif result.confidence_level == ConfidenceLevel.REVIEW_REQUIRED:
            return "VERIFY_MATCH"
        else:
            return "OPTIONAL_REVIEW"

    def get_review_queue(self, summary: MatchingSummary) -> List[Dict]:
        """
        Generate prioritized review queue from summary.

        Args:
            summary: Matching summary with all results

        Returns:
            List of review items sorted by priority
        """
        queue = []

        for result in summary.matches:
            if not result.needs_human_review:
                continue

            priority = self.classifier.get_review_priority(result.confidence_level)

            queue.append({
                "priority": priority,
                "line_item": result.line_item,
                "level": result.confidence_level.value,
                "reason": result.review_reason,
                "action_needed": self._get_action_description(result),
                "matched_sku": result.matched_sku,
                "confidence": result.confidence,
                "quantity": result.quantity,
                "warnings": result.warnings,
            })

        # Sort by priority
        queue.sort(key=lambda x: x["priority"])

        return queue

    def _get_action_description(self, result: MatchResult) -> str:
        """Get human-readable action description."""
        if result.confidence_level == ConfidenceLevel.NEEDS_CLARIFICATION:
            return f"Contact customer for {result.line_item} specifications"
        elif result.confidence_level == ConfidenceLevel.NO_MATCH:
            return f"Source alternative product for {result.line_item}"
        elif result.confidence_level == ConfidenceLevel.REVIEW_REQUIRED:
            return f"Verify match {result.matched_sku} for {result.line_item}"
        else:
            return f"Optional: verify {result.matched_sku}"


def load_requirements_from_file(filepath: str) -> List[Requirement]:
    """
    Load requirements from a JSON file.

    Args:
        filepath: Path to the requirements JSON file

    Returns:
        List of Requirement objects
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    items = data.get("items", [])
    return [Requirement.from_dict(item) for item in items]


def load_requirements_from_dict(data: Dict) -> List[Requirement]:
    """
    Load requirements from a dictionary.

    Args:
        data: Dictionary with 'items' key containing requirements

    Returns:
        List of Requirement objects
    """
    items = data.get("items", [])
    return [Requirement.from_dict(item) for item in items]
