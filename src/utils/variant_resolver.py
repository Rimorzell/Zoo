"""Variant resolution utilities.

Handles base/variant relationships for lighting items
(e.g., L1 is base, L1/B and L1/E are variants).
"""

import re
from typing import List, Dict, Optional, Tuple
from ..models.requirement import Requirement, RequirementSpecs


class VariantResolver:
    """
    Resolves variant relationships and inherits specs from base items.

    Common patterns:
    - L1 (base), L1/B (battery variant), L1/E (emergency generator variant)
    - D1 (base), D1/B (battery), D1/E (emergency)
    """

    # Patterns for identifying variants
    VARIANT_SUFFIXES = ["/B", "/E", "-B", "-E", "_B", "_E"]

    def __init__(self):
        self.base_items: Dict[str, Requirement] = {}

    def resolve_variants(self, requirements: List[Requirement]) -> List[Requirement]:
        """
        Resolve variants and inherit missing specs from base items.

        Args:
            requirements: List of requirements to process

        Returns:
            List of requirements with variants resolved
        """
        # First pass: identify base items
        self._identify_base_items(requirements)

        # Second pass: resolve variants
        resolved = []
        for req in requirements:
            resolved_req = self._resolve_single(req)
            resolved.append(resolved_req)

        return resolved

    def _identify_base_items(self, requirements: List[Requirement]):
        """Identify base items (items without variant suffixes)."""
        self.base_items = {}

        for req in requirements:
            line_item = req.line_item
            base_name = self._get_base_name(line_item)

            # If this is not a variant (line_item equals base_name), it's a base
            if line_item == base_name:
                self.base_items[base_name] = req

    def _get_base_name(self, line_item: str) -> str:
        """Extract base name from line item (remove variant suffix)."""
        for suffix in self.VARIANT_SUFFIXES:
            if line_item.endswith(suffix):
                return line_item[:-len(suffix)]
        return line_item

    def _resolve_single(self, req: Requirement) -> Requirement:
        """Resolve a single requirement's variants."""
        line_item = req.line_item
        base_name = self._get_base_name(line_item)

        # If this is a base item or no base found, return as-is
        if line_item == base_name or base_name not in self.base_items:
            return req

        base_req = self.base_items[base_name]

        # Create new requirement with inherited specs
        return self._inherit_specs(req, base_req)

    def _inherit_specs(self, variant: Requirement, base: Requirement) -> Requirement:
        """
        Inherit missing specs from base item.

        Only inherits specs that are missing in the variant.
        """
        variant_specs = variant.requirements
        base_specs = base.requirements

        # Create list of fields to potentially inherit
        inheritable_fields = [
            "dimensions_mm",
            "wattage",
            "lumens",
            "color_temp_k",
            "reference_manufacturer",
            "reference_catalog",
            "beam_angle",
            "cutout_mm",
        ]

        inherited_notes = []

        for field in inheritable_fields:
            variant_value = getattr(variant_specs, field, None)
            base_value = getattr(base_specs, field, None)

            if variant_value is None and base_value is not None:
                setattr(variant_specs, field, base_value)
                inherited_notes.append(f"Inherited {field} from base item {base.line_item}")

        # Add inheritance notes
        if inherited_notes:
            variant.notes = list(variant.notes) + inherited_notes

            # Boost confidence slightly if we successfully inherited
            if variant.confidence < 0.9:
                variant.confidence = min(0.9, variant.confidence + 0.1)

        return variant

    def get_variant_relationships(self, requirements: List[Requirement]) -> Dict[str, List[str]]:
        """
        Get mapping of base items to their variants.

        Args:
            requirements: List of requirements

        Returns:
            Dictionary mapping base item names to list of variant names
        """
        self._identify_base_items(requirements)

        relationships = {}

        for base_name in self.base_items:
            relationships[base_name] = []

        for req in requirements:
            line_item = req.line_item
            base_name = self._get_base_name(line_item)

            if line_item != base_name and base_name in relationships:
                relationships[base_name].append(line_item)

        # Remove bases with no variants
        return {k: v for k, v in relationships.items() if v}
