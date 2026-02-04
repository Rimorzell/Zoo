"""Variant resolution utilities.

Handles base/variant relationships for lighting items
(e.g., L1 is base, L1/B and L1/E are variants).

Implements the Inheritance Resolution Rule:
- Detect base-derived relationships
- Inherit all constraints from base to derived
- Allow derived to override or extend constraints
- Track inherited vs overridden constraints
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
from copy import deepcopy

from ..models.requirement import Requirement, RequirementSpecs, WattageSpec, LumensSpec, ColorTempSpec
from .inheritance_detector import InheritanceDetector, DetectionResult


@dataclass
class InheritedConstraint:
    """Tracks a single inherited constraint."""
    field_name: str
    value: Any
    source_item: str
    was_overridden: bool = False


@dataclass
class InheritanceInfo:
    """Information about inheritance for a requirement."""
    is_derived: bool = False
    base_item: Optional[str] = None
    inheritance_chain: List[str] = field(default_factory=list)
    inherited_constraints: List[InheritedConstraint] = field(default_factory=list)
    overridden_constraints: List[str] = field(default_factory=list)
    base_confidence: float = 1.0


class VariantResolver:
    """
    Resolves variant relationships and inherits specs from base items.

    Implements the Inheritance Resolution Rule:
    1. Detect base-derived relationships via prefix, reference, or description
    2. Inherit ALL constraints from base (dimensions, product family, performance, environmental)
    3. Derived may override or extend constraints if explicitly stated
    4. Override precedence: derived > base
    5. Track inherited vs overridden for confidence and review routing
    """

    # Patterns for identifying variants
    VARIANT_SUFFIXES = ["/B", "/E", "-B", "-E", "_B", "_E"]

    # All fields that can be inherited from base to derived
    INHERITABLE_FIELDS = [
        # Dimensional constraints
        "dimensions_mm",
        "beam_angle",
        "cutout_mm",
        # Product family constraints
        "product_type",
        # Performance specs
        "wattage",
        "lumens",
        "color_temp_k",
        # Environmental constraints
        "environment",
        "min_ip_rating",
        "mounting",
        # Control features
        "dimmable",
        "dimming_type",
        # Reference info
        "reference_manufacturer",
        "reference_catalog",
    ]

    # Fields that are typically overridden by derived items
    TYPICALLY_OVERRIDDEN = [
        "emergency_type",
        "emergency_duration_hours",
        "mounting",  # e.g., L1W is wall-mounted variant of L1
    ]

    def __init__(self):
        self.base_items: Dict[str, Requirement] = {}
        self.detector = InheritanceDetector()
        self.detection_result: Optional[DetectionResult] = None
        self.inheritance_info: Dict[str, InheritanceInfo] = {}

    def resolve_variants(self, requirements: List[Requirement]) -> List[Requirement]:
        """
        Resolve variants and inherit missing specs from base items.

        Args:
            requirements: List of requirements to process

        Returns:
            List of requirements with variants resolved
        """
        # Step 1: Detect all inheritance relationships
        self.detection_result = self.detector.detect(requirements)

        # Step 2: Build base items map
        self._identify_base_items(requirements)

        # Step 3: Resolve each requirement
        resolved = []
        for req in requirements:
            resolved_req, info = self._resolve_single(req)
            self.inheritance_info[req.line_item] = info
            resolved.append(resolved_req)

        return resolved

    def _identify_base_items(self, requirements: List[Requirement]):
        """Identify base items from requirements."""
        self.base_items = {}

        for req in requirements:
            if self.detection_result and req.line_item in self.detection_result.base_items:
                self.base_items[req.line_item] = req
            elif not self._is_variant(req.line_item):
                # Fallback for items not detected as either base or derived
                self.base_items[req.line_item] = req

    def _is_variant(self, line_item: str) -> bool:
        """Check if item is a variant based on suffix."""
        return any(line_item.endswith(suffix) for suffix in self.VARIANT_SUFFIXES)

    def _get_base_name(self, line_item: str) -> str:
        """Extract base name from line item (remove variant suffix)."""
        for suffix in self.VARIANT_SUFFIXES:
            if line_item.endswith(suffix):
                return line_item[:-len(suffix)]
        return line_item

    def _resolve_single(self, req: Requirement) -> Tuple[Requirement, InheritanceInfo]:
        """
        Resolve a single requirement's variants.

        Returns:
            Tuple of (resolved requirement, inheritance info)
        """
        info = InheritanceInfo()

        # Check if this is a derived item
        if self.detection_result:
            base_item = self.detection_result.derived_items.get(req.line_item)
            if base_item:
                info.is_derived = True
                info.base_item = base_item
                info.inheritance_chain = self.detector.get_inheritance_chain(
                    req.line_item, self.detection_result
                )

        # If not found in detection result, try legacy suffix-based detection
        if not info.is_derived:
            base_name = self._get_base_name(req.line_item)
            if base_name != req.line_item and base_name in self.base_items:
                info.is_derived = True
                info.base_item = base_name
                info.inheritance_chain = [base_name, req.line_item]

        # If this is a base item or no base found, return as-is
        if not info.is_derived or info.base_item not in self.base_items:
            return req, info

        base_req = self.base_items[info.base_item]
        info.base_confidence = base_req.confidence

        # Inherit constraints from base
        resolved_req = self._inherit_constraints(req, base_req, info)

        return resolved_req, info

    def _inherit_constraints(
        self,
        variant: Requirement,
        base: Requirement,
        info: InheritanceInfo
    ) -> Requirement:
        """
        Inherit missing constraints from base item.

        Implements override precedence:
        - If constraint present in derived -> use it (override)
        - If absent -> inherit from base
        - Never require clarification for constraint that exists in base
        """
        variant_specs = variant.requirements
        base_specs = base.requirements

        inherited_notes = []

        for field_name in self.INHERITABLE_FIELDS:
            variant_value = self._get_field_value(variant_specs, field_name)
            base_value = self._get_field_value(base_specs, field_name)

            # Override precedence: if variant has value, use it
            if variant_value is not None:
                # Track as overridden if base also had a value
                if base_value is not None:
                    info.overridden_constraints.append(field_name)
                continue

            # Inherit from base if available
            if base_value is not None:
                self._set_field_value(variant_specs, field_name, base_value)
                inherited_notes.append(f"Inherited {field_name} from {base.line_item}")
                info.inherited_constraints.append(InheritedConstraint(
                    field_name=field_name,
                    value=base_value,
                    source_item=base.line_item,
                    was_overridden=False
                ))

        # Add inheritance notes
        if inherited_notes:
            variant.notes = list(variant.notes) + inherited_notes

        return variant

    def _get_field_value(self, specs: RequirementSpecs, field_name: str) -> Any:
        """Get field value, handling nested objects."""
        value = getattr(specs, field_name, None)

        # For spec objects, check if they have meaningful content
        if isinstance(value, (WattageSpec, LumensSpec)):
            if value.target is None and value.min is None and value.max is None:
                return None
        elif isinstance(value, ColorTempSpec):
            if value.target is None and not value.acceptable:
                return None
        elif isinstance(value, dict):
            if not value or all(v is None for v in value.values()):
                return None

        return value

    def _set_field_value(self, specs: RequirementSpecs, field_name: str, value: Any):
        """Set field value, with deep copy for mutable objects."""
        if isinstance(value, (WattageSpec, LumensSpec, ColorTempSpec, dict, list)):
            value = deepcopy(value)
        setattr(specs, field_name, value)

    def get_inheritance_info(self, line_item: str) -> Optional[InheritanceInfo]:
        """Get inheritance info for a specific item."""
        return self.inheritance_info.get(line_item)

    def get_variant_relationships(self, requirements: List[Requirement]) -> Dict[str, List[str]]:
        """
        Get mapping of base items to their variants.

        Args:
            requirements: List of requirements

        Returns:
            Dictionary mapping base item names to list of variant names
        """
        if self.detection_result is None:
            self.detection_result = self.detector.detect(requirements)

        relationships = {}

        for base_item in self.detection_result.base_items:
            relationships[base_item] = []

        for derived, base in self.detection_result.derived_items.items():
            if base in relationships:
                relationships[base].append(derived)

        # Remove bases with no variants
        return {k: v for k, v in relationships.items() if v}

    def is_constraint_inherited(self, line_item: str, constraint: str) -> bool:
        """
        Check if a specific constraint was inherited for an item.

        Used to determine if clarification is needed.
        """
        info = self.inheritance_info.get(line_item)
        if not info:
            return False

        return any(ic.field_name == constraint for ic in info.inherited_constraints)

    def is_constraint_overridden(self, line_item: str, constraint: str) -> bool:
        """Check if a specific constraint was overridden by derived item."""
        info = self.inheritance_info.get(line_item)
        if not info:
            return False

        return constraint in info.overridden_constraints

    def get_inherited_constraints(self, line_item: str) -> List[str]:
        """Get list of constraints that were inherited for an item."""
        info = self.inheritance_info.get(line_item)
        if not info:
            return []

        return [ic.field_name for ic in info.inherited_constraints]

    def get_overridden_constraints(self, line_item: str) -> List[str]:
        """Get list of constraints that were overridden by the derived item."""
        info = self.inheritance_info.get(line_item)
        if not info:
            return []

        return info.overridden_constraints

    def calculate_derived_confidence(
        self,
        base_confidence: float,
        derived_source_confidence: float
    ) -> float:
        """
        Calculate confidence for a derived match.

        Per the Inheritance Resolution Rule:
        confidence = min(base_confidence, derived_source_confidence)

        Does not downgrade confidence solely due to inheritance.
        """
        return min(base_confidence, derived_source_confidence)

    def needs_clarification(
        self,
        line_item: str,
        missing_constraint: str,
        base_has_constraint: bool
    ) -> bool:
        """
        Determine if clarification is needed for a missing constraint.

        Per the Inheritance Resolution Rule:
        - Do NOT request clarification for inherited constraints
        - Request clarification only if:
          - Constraint is missing in both base and derived, OR
          - Derived overrides conflict with base

        Args:
            line_item: The item to check
            missing_constraint: The constraint that is missing
            base_has_constraint: Whether the base item has this constraint

        Returns:
            True if clarification should be requested
        """
        info = self.inheritance_info.get(line_item)

        # Not a derived item - normal clarification rules apply
        if not info or not info.is_derived:
            return True

        # If base has the constraint, do NOT request clarification
        # (it will be inherited)
        if base_has_constraint:
            return False

        # Missing in both base and derived - need clarification
        return True
