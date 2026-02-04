"""Source validation for lighting requirements.

Validates requirements before matching to ensure they have sufficient
specification data for accurate product matching.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from ..models.requirement import Requirement


class ValidationSeverity(Enum):
    """Severity level of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue found in a requirement."""
    code: str
    message: str
    severity: ValidationSeverity
    field: Optional[str] = None

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "field": self.field,
        }


@dataclass
class ValidationResult:
    """Result of validating a requirement."""
    is_valid: bool
    can_match: bool  # False means NEEDS_CLARIFICATION
    issues: List[ValidationIssue] = field(default_factory=list)
    adjusted_confidence: float = 1.0

    def to_dict(self):
        return {
            "is_valid": self.is_valid,
            "can_match": self.can_match,
            "issues": [i.to_dict() for i in self.issues],
            "adjusted_confidence": self.adjusted_confidence,
        }


class SourceValidator:
    """Validates requirement source data quality and completeness."""

    # Minimum confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.90
    ACCEPTABLE_CONFIDENCE_THRESHOLD = 0.70
    LOW_CONFIDENCE_THRESHOLD = 0.50

    # Product types that require dimension specifications
    DIMENSION_REQUIRED_TYPES = ["waterproof", "linear", "panel"]

    # Red flag patterns in notes
    RED_FLAG_PATTERNS = [
        "inferred",
        "clarify",
        "not found",
        "missing",
        "please clarify",
        "no specification",
        "specs inferred",
    ]

    def validate(self, requirement: Requirement) -> ValidationResult:
        """
        Validate a requirement for matching readiness.

        Args:
            requirement: The requirement to validate

        Returns:
            ValidationResult with issues and match readiness status
        """
        issues: List[ValidationIssue] = []
        can_match = True
        adjusted_confidence = requirement.confidence

        # Check source confidence
        confidence_issues = self._check_confidence(requirement)
        issues.extend(confidence_issues)

        # Check for critical specifications
        spec_issues = self._check_critical_specs(requirement)
        issues.extend(spec_issues)

        # Check notes for red flags
        note_issues = self._check_notes(requirement)
        issues.extend(note_issues)

        # Determine if we can proceed with matching
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]

        if critical_issues:
            can_match = False
            adjusted_confidence = min(adjusted_confidence, 0.3)
        elif error_issues:
            can_match = True  # Can try, but with lower confidence
            adjusted_confidence = min(adjusted_confidence, 0.6)
        elif issues:
            # Warnings reduce confidence slightly
            warning_count = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
            adjusted_confidence = max(0.5, adjusted_confidence - (warning_count * 0.05))

        # Final confidence check
        if requirement.confidence < self.LOW_CONFIDENCE_THRESHOLD:
            can_match = False

        is_valid = len(critical_issues) == 0 and len(error_issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            can_match=can_match,
            issues=issues,
            adjusted_confidence=adjusted_confidence,
        )

    def _check_confidence(self, requirement: Requirement) -> List[ValidationIssue]:
        """Check source confidence score."""
        issues = []

        if requirement.confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            # High confidence - no issues
            pass
        elif requirement.confidence >= self.ACCEPTABLE_CONFIDENCE_THRESHOLD:
            issues.append(ValidationIssue(
                code="MODERATE_CONFIDENCE",
                message=f"Source confidence is moderate ({requirement.confidence:.2f}). "
                        "Some values may have been inferred.",
                severity=ValidationSeverity.INFO,
                field="confidence",
            ))
        elif requirement.confidence >= self.LOW_CONFIDENCE_THRESHOLD:
            issues.append(ValidationIssue(
                code="LOW_CONFIDENCE",
                message=f"Source confidence is low ({requirement.confidence:.2f}). "
                        "Extraction may be unreliable.",
                severity=ValidationSeverity.WARNING,
                field="confidence",
            ))
        else:
            issues.append(ValidationIssue(
                code="VERY_LOW_CONFIDENCE",
                message=f"Source confidence is very low ({requirement.confidence:.2f}). "
                        "Cannot reliably match without clarification.",
                severity=ValidationSeverity.CRITICAL,
                field="confidence",
            ))

        return issues

    def _check_critical_specs(self, requirement: Requirement) -> List[ValidationIssue]:
        """Check for critical specification requirements."""
        issues = []
        specs = requirement.requirements

        # Check for at least one performance specification
        has_wattage = specs.wattage is not None and specs.wattage.target is not None
        has_lumens = specs.lumens is not None and specs.lumens.target is not None
        has_reference = specs.reference_catalog is not None

        if not any([has_wattage, has_lumens, has_reference]):
            issues.append(ValidationIssue(
                code="NO_PERFORMANCE_SPEC",
                message="No performance specifications (wattage, lumens, or reference catalog). "
                        "Cannot determine appropriate product.",
                severity=ValidationSeverity.CRITICAL,
                field="requirements",
            ))

        # Check dimension requirements for linear/panel products
        if specs.product_type in self.DIMENSION_REQUIRED_TYPES:
            if not specs.dimensions_mm:
                issues.append(ValidationIssue(
                    code="MISSING_DIMENSIONS",
                    message=f"Dimension specification required for {specs.product_type} product type "
                            "but not provided.",
                    severity=ValidationSeverity.CRITICAL,
                    field="dimensions_mm",
                ))

        # Check emergency specifications
        if specs.emergency_type == "battery_pack":
            if not specs.emergency_duration_hours:
                issues.append(ValidationIssue(
                    code="MISSING_EMERGENCY_DURATION",
                    message="Battery pack emergency type specified but duration not provided. "
                            "Assuming 3 hours.",
                    severity=ValidationSeverity.WARNING,
                    field="emergency_duration_hours",
                ))

        # Check for product type
        if not specs.product_type:
            issues.append(ValidationIssue(
                code="MISSING_PRODUCT_TYPE",
                message="Product type not specified. Cannot determine compatible product families.",
                severity=ValidationSeverity.ERROR,
                field="product_type",
            ))

        return issues

    def _check_notes(self, requirement: Requirement) -> List[ValidationIssue]:
        """Check notes for red flags indicating extraction issues."""
        issues = []

        for note in requirement.notes:
            note_lower = note.lower()

            # Check for red flag patterns
            for pattern in self.RED_FLAG_PATTERNS:
                if pattern in note_lower:
                    if "please clarify" in note_lower or "no specification" in note_lower:
                        issues.append(ValidationIssue(
                            code="CLARIFICATION_NEEDED",
                            message=f"Extraction note indicates clarification needed: {note}",
                            severity=ValidationSeverity.ERROR,
                            field="notes",
                        ))
                    elif "inferred" in note_lower:
                        issues.append(ValidationIssue(
                            code="INFERRED_DATA",
                            message=f"Some data was inferred: {note}",
                            severity=ValidationSeverity.WARNING,
                            field="notes",
                        ))
                    else:
                        issues.append(ValidationIssue(
                            code="EXTRACTION_WARNING",
                            message=f"Extraction warning: {note}",
                            severity=ValidationSeverity.INFO,
                            field="notes",
                        ))
                    break  # Only add one issue per note

        return issues

    def batch_validate(self, requirements: List[Requirement]) -> dict:
        """
        Validate multiple requirements and return summary.

        Args:
            requirements: List of requirements to validate

        Returns:
            Dictionary with validation summary and per-item results
        """
        results = {}
        summary = {
            "total": len(requirements),
            "valid": 0,
            "can_match": 0,
            "needs_clarification": 0,
            "issues_by_code": {},
        }

        for req in requirements:
            result = self.validate(req)
            results[req.line_item] = result

            if result.is_valid:
                summary["valid"] += 1
            if result.can_match:
                summary["can_match"] += 1
            else:
                summary["needs_clarification"] += 1

            # Count issues by code
            for issue in result.issues:
                if issue.code not in summary["issues_by_code"]:
                    summary["issues_by_code"][issue.code] = 0
                summary["issues_by_code"][issue.code] += 1

        return {
            "summary": summary,
            "results": {k: v.to_dict() for k, v in results.items()},
        }
