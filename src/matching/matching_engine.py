"""Main matching engine orchestrator.

Coordinates all components to match requirements to products.
Implements inheritance-aware matching for base-derived relationships.
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
from ..utils.variant_resolver import VariantResolver, InheritanceInfo
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
    enable_relaxed_fallback: bool = True
    relaxed_dimension_tolerance: float = 0.20
    relaxed_ip_tolerance: int = 2

    # Selection options
    use_llm_selection: bool = True
    llm_backend: Optional[Any] = None

    # Output options
    include_all_candidates: bool = False
    max_alternatives: int = 3

    # Inheritance options
    use_inheritance_resolution: bool = True
    prefer_explicit_sku_features: bool = True


class MatchingEngine:
    """
    Main engine for matching lighting requirements to products.

    Orchestrates the full matching pipeline with inheritance support:
    1. Variant resolution (detect base-derived relationships)
    2. Source validation
    3. Hard constraint filtering
    4. Soft preference scoring
    5. LLM/rule-based selection (with SKU feature preference)
    6. Confidence classification
    7. Result generation with inheritance tracking
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
        self.variant_resolver = VariantResolver()

        # Product catalog
        self.products: List[Product] = []

        # Cached base matches for inheritance
        self._base_matches: Dict[str, MatchResult] = {}

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
        validation_result: Optional[ValidationResult] = None,
        inheritance_info: Optional[InheritanceInfo] = None
    ) -> MatchResult:
        """
        Match a single requirement to the best product.

        Args:
            requirement: The requirement to match
            validation_result: Pre-computed validation result (optional)
            inheritance_info: Inheritance info from variant resolution (optional)

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

        # Check if this is a derived item with inheritance
        is_derived = inheritance_info and inheritance_info.is_derived
        base_result = None

        if is_derived and inheritance_info.base_item:
            # Get cached base match if available
            base_result = self._base_matches.get(inheritance_info.base_item)

            if base_result:
                result.is_derived_match = True
                result.base_item = inheritance_info.base_item
                result.base_sku = base_result.matched_sku
                result.inherited_constraints = self.variant_resolver.get_inherited_constraints(
                    requirement.line_item
                )
                result.overridden_constraints = self.variant_resolver.get_overridden_constraints(
                    requirement.line_item
                )

        # Step 1: Validate source if not provided
        if validation_result is None and not self.config.skip_validation:
            validation_result = self.validator.validate(requirement)

        # Check if we can proceed with matching
        if validation_result and not validation_result.can_match:
            # For derived items, check if we can use inherited constraints
            if is_derived and base_result and base_result.matched_sku:
                # Don't require clarification for inherited constraints
                critical_issues = [
                    i for i in validation_result.issues
                    if i.severity.value == "critical" and
                    not self._is_inherited_constraint_issue(i.message, result.inherited_constraints)
                ]
                if not critical_issues:
                    # Can proceed with inherited constraints
                    validation_result = None

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

        # For derived items with overridden constraints, use inheritance-aware matching
        if is_derived and base_result and result.overridden_constraints:
            return self._match_derived_with_overrides(
                requirement, result, base_result, inheritance_info, validation_result
            )

        # Standard matching flow
        return self._match_standard(requirement, result, validation_result, inheritance_info)

    def _match_standard(
        self,
        requirement: Requirement,
        result: MatchResult,
        validation_result: Optional[ValidationResult],
        inheritance_info: Optional[InheritanceInfo]
    ) -> MatchResult:
        """Standard matching flow for base items and derived items without overrides."""

        # Step 2: Filter products by hard constraints
        filter_result = self.filter.filter(
            requirement,
            self.products,
            dimension_tolerance_override=(
                None if self.config.strict_dimensions else self.config.relaxed_dimension_tolerance
            ),
            ip_tolerance_override=(None if self.config.allow_ip_tolerance else 0),
        )
        result.candidates_evaluated = len(filter_result.candidates)

        if not filter_result.candidates and self.config.enable_relaxed_fallback:
            filter_result = self.filter.filter(
                requirement,
                self.products,
                dimension_tolerance_override=self.config.relaxed_dimension_tolerance,
                ip_tolerance_override=self.config.relaxed_ip_tolerance,
                allow_missing_ip=True,
            )
            result.candidates_evaluated = len(filter_result.candidates)
            if filter_result.candidates:
                result.warnings.append(
                    "No candidates met strict constraints; used relaxed fallback matching."
                )

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

        # Apply SKU feature preference if enabled
        if self.config.prefer_explicit_sku_features:
            scored_candidates = self._apply_sku_feature_preference(
                requirement, scored_candidates
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

        # Step 5: Apply confidence classification
        result = self.classifier.apply_to_result(result, validation_result)

        # Step 6: Apply inheritance-specific adjustments
        if inheritance_info and inheritance_info.is_derived:
            result = self._apply_inheritance_confidence(result, inheritance_info)

        # Add constraint checks
        if filter_result.constraint_checks:
            result.constraint_checks = filter_result.constraint_checks[:6]

        # Set confidence reason
        if result.is_derived_match:
            result.confidence_reason = "DERIVED_FROM_BASE"
        else:
            result.confidence_reason = "DIRECT_MATCH"

        return result

    def _match_derived_with_overrides(
        self,
        requirement: Requirement,
        result: MatchResult,
        base_result: MatchResult,
        inheritance_info: InheritanceInfo,
        validation_result: Optional[ValidationResult]
    ) -> MatchResult:
        """
        Match a derived item by applying overrides to the base match.

        Per Inheritance Resolution Rule:
        - Apply derived overrides to the selected SKU
        - Re-evaluate only overridden constraints, not inherited ones
        """
        # Start with the base SKU and re-evaluate with overrides

        # Filter products, but we want to prefer variants of the base product
        filter_result = self.filter.filter(
            requirement,
            self.products,
            dimension_tolerance_override=(
                None if self.config.strict_dimensions else self.config.relaxed_dimension_tolerance
            ),
            ip_tolerance_override=(None if self.config.allow_ip_tolerance else 0),
        )
        result.candidates_evaluated = len(filter_result.candidates)

        if not filter_result.candidates and self.config.enable_relaxed_fallback:
            filter_result = self.filter.filter(
                requirement,
                self.products,
                dimension_tolerance_override=self.config.relaxed_dimension_tolerance,
                ip_tolerance_override=self.config.relaxed_ip_tolerance,
                allow_missing_ip=True,
            )
            result.candidates_evaluated = len(filter_result.candidates)
            if filter_result.candidates:
                result.warnings.append(
                    "No candidates met strict constraints; used relaxed fallback matching."
                )

        if not filter_result.candidates:
            result.confidence_level = ConfidenceLevel.NO_MATCH
            result.reasoning = self._build_no_match_reasoning(filter_result)
            result.needs_human_review = True
            result.review_reason = "No products match derived requirements"
            result.warnings = [
                f"Eliminated {p.name}: {reason}"
                for p, reason in filter_result.eliminated[:5]
            ]
            return result

        # Score candidates
        scored_candidates = self.scorer.score_candidates(
            requirement, filter_result.candidates
        )

        # Apply SKU feature preference - especially important for derived items
        # with emergency/mounting overrides
        if self.config.prefer_explicit_sku_features:
            scored_candidates = self._apply_sku_feature_preference(
                requirement, scored_candidates
            )

        # Prefer products that are variants of the base product
        scored_candidates = self._prefer_base_variants(
            scored_candidates, base_result.matched_sku
        )

        # Select best match
        if len(scored_candidates) == 1:
            result.selection_method = "single_candidate"
            best = scored_candidates[0]
            result.matched_sku = best.product.sku
            result.matched_product_name = best.product.name
            result.confidence = best.score
            result.reasoning = f"Derived match from {result.base_item}: {best.product.name}"
            result.constraint_checks = filter_result.constraint_checks
            result.warnings = best.notes

        else:
            if self.config.use_llm_selection and self.selector.needs_llm_selection(scored_candidates):
                result.selection_method = "llm_selection"
            else:
                result.selection_method = "rule_based_selection"

            selection = self.selector.select(requirement, scored_candidates)

            result.matched_sku = selection.selected_sku
            result.matched_product_name = selection.selected_name
            result.confidence = selection.confidence
            result.reasoning = f"Derived match from {result.base_item}: {selection.reasoning}"
            result.warnings = selection.warnings

            for alt in selection.alternatives[:self.config.max_alternatives]:
                result.alternatives.append(Alternative(
                    sku=alt["sku"],
                    name=alt.get("name", ""),
                    confidence=alt["confidence"],
                    trade_off=alt.get("note", ""),
                ))

        # Apply confidence classification
        result = self.classifier.apply_to_result(result, validation_result)

        # Apply inheritance-specific confidence
        result = self._apply_inheritance_confidence(result, inheritance_info)

        # Set confidence reason
        result.confidence_reason = "DERIVED_FROM_BASE"

        # Add constraint checks
        if filter_result.constraint_checks:
            result.constraint_checks = filter_result.constraint_checks[:6]

        return result

    def _apply_sku_feature_preference(
        self,
        requirement: Requirement,
        scored_candidates: List[ScoredCandidate]
    ) -> List[ScoredCandidate]:
        """
        Apply SKU preference for explicitly encoded features.

        When multiple SKUs satisfy constraints, prefer SKUs that explicitly
        encode required features like emergency, mounting, IP rating.
        """
        if not scored_candidates:
            return scored_candidates

        emergency_type = requirement.requirements.emergency_type
        mounting = requirement.requirements.mounting
        ip_rating = requirement.requirements.min_ip_rating

        for candidate in scored_candidates:
            sku = candidate.product.sku.upper()
            bonus = 0.0

            # Emergency feature preference
            if emergency_type:
                if emergency_type == "battery_pack":
                    # Prefer SKUs with -EM, -B, -BAT, etc.
                    if any(marker in sku for marker in ["-EM", "-B-", "-BAT", "-EMG", "EM-"]):
                        bonus += 0.03
                        if "EM" not in candidate.notes:
                            candidate.notes.append("SKU explicitly encodes emergency feature")
                elif emergency_type == "central_generator":
                    # Standard products are fine, no bonus needed
                    pass

            # Mounting preference
            if mounting:
                mounting_markers = {
                    "surface": ["-SMD", "-SF", "-SUR", "-SM"],
                    "recessed": ["-REC", "-RC", "-EMB"],
                    "wall": ["-WL", "-WM", "-WALL"],
                    "suspended": ["-SUS", "-PEN", "-SP"],
                }
                markers = mounting_markers.get(mounting.lower(), [])
                if any(marker in sku for marker in markers):
                    bonus += 0.02
                    if "mounting" not in " ".join(candidate.notes).lower():
                        candidate.notes.append(f"SKU explicitly encodes {mounting} mounting")

            # IP rating preference
            if ip_rating:
                ip_str = f"IP{ip_rating}"
                if ip_str in sku:
                    bonus += 0.02
                    if "IP" not in " ".join(candidate.notes):
                        candidate.notes.append(f"SKU explicitly encodes {ip_str}")

            # Apply bonus
            if bonus > 0:
                candidate.score = min(1.0, candidate.score + bonus)

        # Re-sort by score
        scored_candidates.sort(key=lambda x: x.score, reverse=True)

        return scored_candidates

    def _prefer_base_variants(
        self,
        scored_candidates: List[ScoredCandidate],
        base_sku: Optional[str]
    ) -> List[ScoredCandidate]:
        """Prefer products that are variants of the base product SKU."""
        if not base_sku or not scored_candidates:
            return scored_candidates

        # Extract base SKU prefix (before the variant suffix)
        base_prefix = base_sku.rsplit("-", 1)[0] if "-" in base_sku else base_sku

        for candidate in scored_candidates:
            sku = candidate.product.sku
            # Check if this SKU is a variant of the base
            if sku.startswith(base_prefix) or base_prefix in sku:
                candidate.score = min(1.0, candidate.score + 0.05)
                if "base variant" not in " ".join(candidate.notes).lower():
                    candidate.notes.append(f"Product is variant of base SKU {base_sku}")

        # Re-sort by score
        scored_candidates.sort(key=lambda x: x.score, reverse=True)

        return scored_candidates

    def _apply_inheritance_confidence(
        self,
        result: MatchResult,
        inheritance_info: InheritanceInfo
    ) -> MatchResult:
        """
        Apply inheritance-specific confidence adjustments.

        Per Inheritance Resolution Rule:
        confidence = min(base_confidence, derived_source_confidence)
        Do not downgrade confidence solely due to inheritance.
        """
        if not inheritance_info.is_derived:
            return result

        # Calculate confidence per the rule
        base_confidence = inheritance_info.base_confidence
        derived_confidence = result.source_confidence

        # Use minimum of base and derived confidence
        inherited_confidence = min(base_confidence, derived_confidence)

        # Don't downgrade match confidence solely due to inheritance
        # Only apply if it would improve or maintain confidence
        if result.overridden_constraints:
            result.confidence = min(result.confidence, inherited_confidence)

        # Check if overridden constraints introduce ambiguity
        if result.overridden_constraints:
            # If there are many overrides, reduce confidence slightly
            if len(result.overridden_constraints) > 2:
                result.confidence = min(result.confidence, 0.90)
                if "Multiple overridden constraints" not in result.warnings:
                    result.warnings.append(
                        f"Multiple overridden constraints ({len(result.overridden_constraints)}): "
                        + ", ".join(result.overridden_constraints)
                    )

        # Do not escalate to human review unless overridden constraints introduce ambiguity
        if result.confidence_level in [ConfidenceLevel.REVIEW_REQUIRED, ConfidenceLevel.REVIEW_RECOMMENDED]:
            # Check if the review is only due to inheritance
            if not result.overridden_constraints or len(result.overridden_constraints) <= 2:
                # Only inherited constraints - don't require review
                if result.confidence >= 0.85:
                    result.confidence_level = ConfidenceLevel.AUTO_APPROVE
                    result.needs_human_review = False
                    result.review_reason = None

        return result

    def _is_inherited_constraint_issue(
        self,
        issue_message: str,
        inherited_constraints: List[str]
    ) -> bool:
        """Check if a validation issue is about an inherited constraint."""
        issue_lower = issue_message.lower()
        for constraint in inherited_constraints:
            if constraint.lower() in issue_lower:
                return True
        return False

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

        # Step 0: Resolve variants and inheritance
        if self.config.use_inheritance_resolution:
            requirements = self.variant_resolver.resolve_variants(requirements)

        # Clear cached base matches
        self._base_matches = {}

        # Batch validate first
        validation_results = {}
        if not self.config.skip_validation:
            for req in requirements:
                validation_results[req.line_item] = self.validator.validate(req)

        # Match base items first, then derived items
        base_items = []
        derived_items = []

        for req in requirements:
            info = self.variant_resolver.get_inheritance_info(req.line_item)
            if info and info.is_derived:
                derived_items.append((req, info))
            else:
                base_items.append((req, info))

        # Match base items first
        for req, info in base_items:
            validation = validation_results.get(req.line_item)
            result = self.match_single(req, validation, info)
            summary.matches.append(result)

            # Cache base match for derived items
            self._base_matches[req.line_item] = result

            self._update_summary_counters(summary, result)

        # Match derived items (can use cached base matches)
        for req, info in derived_items:
            validation = validation_results.get(req.line_item)
            result = self.match_single(req, validation, info)
            summary.matches.append(result)

            self._update_summary_counters(summary, result)

        return summary

    def _update_summary_counters(self, summary: MatchingSummary, result: MatchResult):
        """Update summary counters for a match result."""
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
            if result.is_derived_match:
                action["is_derived"] = True
                action["base_item"] = result.base_item
            summary.action_required.append(action)

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

            item = {
                "priority": priority,
                "line_item": result.line_item,
                "level": result.confidence_level.value,
                "reason": result.review_reason,
                "action_needed": self._get_action_description(result),
                "matched_sku": result.matched_sku,
                "confidence": result.confidence,
                "quantity": result.quantity,
                "warnings": result.warnings,
            }

            # Add inheritance info
            if result.is_derived_match:
                item["is_derived"] = True
                item["base_item"] = result.base_item
                item["confidence_reason"] = result.confidence_reason

            queue.append(item)

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

    def get_inheritance_summary(self) -> Dict:
        """Get summary of inheritance relationships detected."""
        if not self.variant_resolver.detection_result:
            return {"relationships": {}, "base_items": [], "derived_items": {}}

        result = self.variant_resolver.detection_result
        return {
            "relationships": {
                base: {
                    "derived_items": rel.derived_items,
                    "relationship_type": rel.relationship_type,
                    "confidence": rel.confidence,
                }
                for base, rel in result.relationships.items()
            },
            "base_items": list(result.base_items),
            "derived_items": dict(result.derived_items),
            "orphan_items": list(result.orphan_items),
        }


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
