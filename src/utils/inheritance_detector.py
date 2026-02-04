"""Inheritance detection for item relationships.

Detects base-derived relationships between items based on:
- Common identifier prefix
- Shared reference product/catalog
- Description that explicitly references another item
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from ..models.requirement import Requirement


@dataclass
class InheritanceRelationship:
    """Represents a base-derived relationship between items."""
    base_item: str
    derived_items: List[str] = field(default_factory=list)
    relationship_type: str = "prefix"  # "prefix", "reference", "description"
    confidence: float = 1.0


@dataclass
class DetectionResult:
    """Result of inheritance detection across all items."""
    relationships: Dict[str, InheritanceRelationship] = field(default_factory=dict)
    base_items: Set[str] = field(default_factory=set)
    derived_items: Dict[str, str] = field(default_factory=dict)  # derived -> base mapping
    orphan_items: Set[str] = field(default_factory=set)


class InheritanceDetector:
    """
    Detects base-derived relationships between requirement items.

    Detection strategies:
    1. Prefix-based: L1 -> L1/B, L1/E, L1W (common prefix with variant suffixes)
    2. Reference-based: Items sharing the same reference_catalog are related
    3. Description-based: "L1W Fixture" explicitly references L1W as base
    """

    # Variant suffixes that indicate a derived item
    VARIANT_SUFFIXES = ["/B", "/E", "-B", "-E", "_B", "_E"]

    # Patterns that indicate item references in descriptions
    DESCRIPTION_REFERENCE_PATTERNS = [
        r"^(\w+)\s+(?:Fixture|with|fed\s+from)",  # "L1 Fixture with..."
        r"Type\s+(\w+)",  # "Type L1W"
        r"Same\s+as\s+(\w+)",  # "Same as L1"
        r"Based\s+on\s+(\w+)",  # "Based on L1"
        r"Variant\s+of\s+(\w+)",  # "Variant of L1"
    ]

    def detect(self, requirements: List[Requirement]) -> DetectionResult:
        """
        Detect all inheritance relationships in the requirements.

        Args:
            requirements: List of requirements to analyze

        Returns:
            DetectionResult with all detected relationships
        """
        result = DetectionResult()
        item_map = {req.line_item: req for req in requirements}

        # Strategy 1: Prefix-based detection
        prefix_relationships = self._detect_by_prefix(requirements)

        # Strategy 2: Reference-based detection
        reference_relationships = self._detect_by_reference(requirements)

        # Strategy 3: Description-based detection
        description_relationships = self._detect_by_description(requirements, item_map)

        # Merge relationships (prefix takes precedence, then reference, then description)
        all_relationships = {}

        # Add description relationships first (lowest priority)
        for base_item, relationship in description_relationships.items():
            all_relationships[base_item] = relationship

        # Add reference relationships (medium priority)
        for base_item, relationship in reference_relationships.items():
            if base_item in all_relationships:
                # Merge derived items
                existing = all_relationships[base_item]
                for derived in relationship.derived_items:
                    if derived not in existing.derived_items:
                        existing.derived_items.append(derived)
            else:
                all_relationships[base_item] = relationship

        # Add prefix relationships (highest priority)
        for base_item, relationship in prefix_relationships.items():
            if base_item in all_relationships:
                # Merge derived items
                existing = all_relationships[base_item]
                for derived in relationship.derived_items:
                    if derived not in existing.derived_items:
                        existing.derived_items.append(derived)
                existing.relationship_type = "prefix"  # Upgrade type
            else:
                all_relationships[base_item] = relationship

        # Build result
        result.relationships = all_relationships

        for base_item, relationship in all_relationships.items():
            result.base_items.add(base_item)
            for derived in relationship.derived_items:
                result.derived_items[derived] = base_item

        # Find orphan items (neither base nor derived)
        all_items = set(item_map.keys())
        classified = result.base_items | set(result.derived_items.keys())
        result.orphan_items = all_items - classified

        return result

    def _detect_by_prefix(self, requirements: List[Requirement]) -> Dict[str, InheritanceRelationship]:
        """
        Detect relationships based on common identifier prefixes.

        Examples:
        - L1 is base for L1/B, L1/E
        - L1W is base for L1W/B, L1W/E (L1W itself derived from L1)
        """
        relationships = {}
        line_items = [req.line_item for req in requirements]

        # Find potential base items (items that don't have variant suffixes)
        potential_bases = []
        for item in line_items:
            is_variant = any(item.endswith(suffix) for suffix in self.VARIANT_SUFFIXES)
            if not is_variant:
                potential_bases.append(item)

        # For each potential base, find derived items
        for base in potential_bases:
            derived = []

            for item in line_items:
                if item == base:
                    continue

                # Check if item is a direct variant of base (base + suffix)
                for suffix in self.VARIANT_SUFFIXES:
                    if item == base + suffix:
                        derived.append(item)
                        break

                # Check if item starts with base and has additional characters
                # But is NOT just base + alphanumeric (which would be a different item like L1W from L1)
                # We want L1/B but not L1W (L1W is its own item)

            if derived:
                relationships[base] = InheritanceRelationship(
                    base_item=base,
                    derived_items=derived,
                    relationship_type="prefix",
                    confidence=1.0
                )

        # Handle nested hierarchies (L1 -> L1W -> L1W/B)
        # L1W is derived from L1 but also a base for L1W/B
        for item in line_items:
            # Check if this item (without variant suffix) could be derived from a shorter base
            base_item = self._strip_variant_suffix(item)
            if base_item == item:
                # No suffix, check if it's a modification of a shorter item
                # e.g., L1W might be related to L1
                potential_parent = self._find_parent_base(base_item, potential_bases)
                if potential_parent and potential_parent != base_item:
                    # Create relationship if not exists
                    if potential_parent not in relationships:
                        relationships[potential_parent] = InheritanceRelationship(
                            base_item=potential_parent,
                            derived_items=[],
                            relationship_type="prefix",
                            confidence=1.0
                        )
                    if base_item not in relationships[potential_parent].derived_items:
                        relationships[potential_parent].derived_items.append(base_item)

        return relationships

    def _strip_variant_suffix(self, item: str) -> str:
        """Remove variant suffix from item name."""
        for suffix in self.VARIANT_SUFFIXES:
            if item.endswith(suffix):
                return item[:-len(suffix)]
        return item

    def _find_parent_base(self, item: str, bases: List[str]) -> Optional[str]:
        """
        Find if item could be a modification of an existing base.

        E.g., L1W could be related to L1 (wall-mounted variant)
        """
        # Sort bases by length (longest first) to find most specific match
        sorted_bases = sorted(bases, key=len, reverse=True)

        for base in sorted_bases:
            if item.startswith(base) and item != base:
                # Check if the remaining part is a valid modifier
                remainder = item[len(base):]
                # Valid modifiers: W (wall), C (ceiling), S (surface), single letters/numbers
                if len(remainder) <= 2 and remainder.isalnum():
                    return base

        return None

    def _detect_by_reference(self, requirements: List[Requirement]) -> Dict[str, InheritanceRelationship]:
        """
        Detect relationships based on shared reference catalog.

        Items with the same reference_catalog are likely related.
        """
        relationships = {}

        # Group items by reference catalog
        by_catalog: Dict[str, List[Requirement]] = {}
        for req in requirements:
            catalog = req.requirements.reference_catalog
            if catalog:
                if catalog not in by_catalog:
                    by_catalog[catalog] = []
                by_catalog[catalog].append(req)

        # For each group, find the base (typically the one without variant suffix)
        for catalog, reqs in by_catalog.items():
            if len(reqs) < 2:
                continue

            # Find base item (no variant suffix, or shortest name)
            base_req = None
            for req in reqs:
                is_variant = any(req.line_item.endswith(s) for s in self.VARIANT_SUFFIXES)
                if not is_variant:
                    if base_req is None or len(req.line_item) < len(base_req.line_item):
                        base_req = req

            if base_req is None:
                # All are variants, pick shortest as pseudo-base
                base_req = min(reqs, key=lambda r: len(r.line_item))

            derived = [r.line_item for r in reqs if r.line_item != base_req.line_item]

            if derived:
                relationships[base_req.line_item] = InheritanceRelationship(
                    base_item=base_req.line_item,
                    derived_items=derived,
                    relationship_type="reference",
                    confidence=0.9
                )

        return relationships

    def _detect_by_description(
        self,
        requirements: List[Requirement],
        item_map: Dict[str, Requirement]
    ) -> Dict[str, InheritanceRelationship]:
        """
        Detect relationships based on description references.

        E.g., "L1W Fixture with Battery" references L1W as base.
        """
        relationships = {}

        for req in requirements:
            description = req.description

            for pattern in self.DESCRIPTION_REFERENCE_PATTERNS:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    referenced_item = match.group(1)

                    # Check if referenced item exists and is different
                    if referenced_item in item_map and referenced_item != req.line_item:
                        if referenced_item not in relationships:
                            relationships[referenced_item] = InheritanceRelationship(
                                base_item=referenced_item,
                                derived_items=[],
                                relationship_type="description",
                                confidence=0.8
                            )

                        if req.line_item not in relationships[referenced_item].derived_items:
                            relationships[referenced_item].derived_items.append(req.line_item)
                        break

        return relationships

    def get_base_for_item(self, item: str, result: DetectionResult) -> Optional[str]:
        """Get the base item for a given derived item."""
        return result.derived_items.get(item)

    def is_derived_item(self, item: str, result: DetectionResult) -> bool:
        """Check if an item is derived from another."""
        return item in result.derived_items

    def is_base_item(self, item: str, result: DetectionResult) -> bool:
        """Check if an item is a base item."""
        return item in result.base_items

    def get_inheritance_chain(self, item: str, result: DetectionResult) -> List[str]:
        """
        Get the full inheritance chain for an item (base -> intermediate -> item).

        Returns list from root base to the item itself.
        """
        chain = [item]
        current = item

        while current in result.derived_items:
            base = result.derived_items[current]
            chain.insert(0, base)
            current = base

        return chain
