"""Confidence classification for match results.

Classifies matches into categories that determine human review requirements.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..models.match_result import ConfidenceLevel, MatchResult
from ..validation.source_validator import ValidationResult


class ConfidenceClassifier:
    """Classifies match confidence and determines review requirements."""

    # Confidence thresholds
    AUTO_APPROVE_THRESHOLD = 0.90
    REVIEW_RECOMMENDED_THRESHOLD = 0.70
    REVIEW_REQUIRED_THRESHOLD = 0.50

    def classify(
        self,
        confidence_score: float,
        validation_result: Optional[ValidationResult] = None,
        candidates_count: int = 0,
        has_warnings: bool = False,
        selection_method: str = ""
    ) -> Tuple[ConfidenceLevel, bool, Optional[str]]:
        """
        Classify confidence and determine review requirements.

        Args:
            confidence_score: The match confidence score (0-1)
            validation_result: Source validation result if available
            candidates_count: Number of candidates that passed filtering
            has_warnings: Whether there are warnings about the match
            selection_method: How the match was selected

        Returns:
            Tuple of (ConfidenceLevel, needs_human_review, review_reason)
        """
        # Check for clarification needed (from validation)
        if validation_result and not validation_result.can_match:
            return (
                ConfidenceLevel.NEEDS_CLARIFICATION,
                True,
                "Source data requires clarification before matching"
            )

        # Check for no match
        if candidates_count == 0:
            return (
                ConfidenceLevel.NO_MATCH,
                True,
                "No products in catalog match the requirements"
            )

        # Adjust confidence based on factors
        adjusted_confidence = confidence_score

        # Reduce confidence if there are warnings
        if has_warnings:
            adjusted_confidence = min(adjusted_confidence, 0.85)

        # Boost confidence for single candidate matches
        if candidates_count == 1 and selection_method == "single_candidate":
            adjusted_confidence = min(1.0, adjusted_confidence + 0.05)

        # Classify based on adjusted confidence
        if adjusted_confidence >= self.AUTO_APPROVE_THRESHOLD:
            return (
                ConfidenceLevel.AUTO_APPROVE,
                False,
                None
            )
        elif adjusted_confidence >= self.REVIEW_RECOMMENDED_THRESHOLD:
            reason = self._get_review_reason(
                adjusted_confidence,
                candidates_count,
                has_warnings,
                selection_method
            )
            return (
                ConfidenceLevel.REVIEW_RECOMMENDED,
                True,  # Flag for review but not required
                reason
            )
        elif adjusted_confidence >= self.REVIEW_REQUIRED_THRESHOLD:
            reason = self._get_review_reason(
                adjusted_confidence,
                candidates_count,
                has_warnings,
                selection_method
            )
            return (
                ConfidenceLevel.REVIEW_REQUIRED,
                True,
                reason
            )
        else:
            return (
                ConfidenceLevel.REVIEW_REQUIRED,
                True,
                "Very low confidence match requires verification"
            )

    def _get_review_reason(
        self,
        confidence: float,
        candidates_count: int,
        has_warnings: bool,
        selection_method: str
    ) -> str:
        """Generate appropriate review reason."""
        reasons = []

        if confidence < 0.75:
            reasons.append("low match confidence")

        if candidates_count > 3:
            reasons.append("multiple similar candidates")

        if has_warnings:
            reasons.append("specification warnings present")

        if selection_method == "llm_selection":
            reasons.append("required intelligent selection between candidates")

        if not reasons:
            reasons.append("moderate confidence level")

        return "Review recommended: " + ", ".join(reasons)

    def apply_to_result(
        self,
        result: MatchResult,
        validation_result: Optional[ValidationResult] = None
    ) -> MatchResult:
        """
        Apply classification to an existing match result.

        Args:
            result: The match result to classify
            validation_result: Optional validation result

        Returns:
            Updated MatchResult with classification applied
        """
        level, needs_review, reason = self.classify(
            confidence_score=result.confidence,
            validation_result=validation_result,
            candidates_count=result.candidates_evaluated,
            has_warnings=len(result.warnings) > 0,
            selection_method=result.selection_method,
        )

        result.confidence_level = level
        result.needs_human_review = needs_review
        result.review_reason = reason

        return result

    def get_review_priority(self, level: ConfidenceLevel) -> int:
        """Get priority order for human review queue (lower = higher priority)."""
        priority_map = {
            ConfidenceLevel.NEEDS_CLARIFICATION: 1,
            ConfidenceLevel.NO_MATCH: 2,
            ConfidenceLevel.REVIEW_REQUIRED: 3,
            ConfidenceLevel.REVIEW_RECOMMENDED: 4,
            ConfidenceLevel.AUTO_APPROVE: 5,
        }
        return priority_map.get(level, 5)

    def summarize_classifications(
        self,
        results: List[MatchResult]
    ) -> dict:
        """
        Generate summary of classifications across all results.

        Args:
            results: List of match results

        Returns:
            Summary dictionary
        """
        summary = {
            "total": len(results),
            "by_level": {},
            "needs_review": 0,
            "auto_approved": 0,
            "high_priority_review": [],
        }

        for result in results:
            level = result.confidence_level.value
            if level not in summary["by_level"]:
                summary["by_level"][level] = 0
            summary["by_level"][level] += 1

            if result.needs_human_review:
                summary["needs_review"] += 1
            else:
                summary["auto_approved"] += 1

            # Track high priority items
            priority = self.get_review_priority(result.confidence_level)
            if priority <= 3:
                summary["high_priority_review"].append({
                    "line_item": result.line_item,
                    "level": level,
                    "reason": result.review_reason,
                    "priority": priority,
                })

        # Sort high priority by priority value
        summary["high_priority_review"].sort(key=lambda x: x["priority"])

        return summary
