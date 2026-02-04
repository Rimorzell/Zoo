"""Hard constraint filtering for product matching.

Eliminates products that cannot possibly match the requirements
based on hard constraints like dimensions, IP rating, etc.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import re

from ..models.requirement import Requirement
from ..models.product import Product
from ..models.match_result import ConstraintCheck


@dataclass
class FilterResult:
    """Result of filtering products against a requirement."""
    candidates: List[Product]
    eliminated: List[Tuple[Product, str]]  # (product, reason)
    constraint_checks: List[ConstraintCheck]


class ConstraintFilter:
    """Filters products based on hard constraints."""

    # Product family compatibility matrix
    FAMILY_COMPATIBILITY = {
        "downlight": [
            "downlight", "downlight_fixed", "downlight_adjustable",
            "downlight_gimbal", "spotlight_recessed"
        ],
        "panel": [
            "panel", "backlit_panel", "edgelit_panel",
            "troffer", "lay_in_fixture"
        ],
        "waterproof": [
            "waterproof_batten", "waterproof_linear",
            "vapor_tight", "wet_location_linear"
        ],
        "linear": [
            "led_strip", "linear_profile", "cove_light",
            "tape_light", "flexible_strip"
        ],
        "wall": [
            "wall_pack", "wall_sconce", "bulkhead",
            "wall_washer", "picture_light", "wall"
        ],
        "highbay": [
            "high_bay", "low_bay", "industrial_pendant",
            "warehouse_light"
        ],
        "flood": [
            "floodlight", "projector", "area_light",
            "security_light", "obstruction_light"
        ],
        "exit": [
            "exit_sign", "emergency_exit", "combo_exit"
        ],
        "emergency": [
            "emergency_light", "emergency_pack",
            "emergency_batten"
        ],
        "track": [
            "track_head", "track_pendant", "track_spot"
        ],
        "pendant": [
            "pendant", "chandelier", "suspended_linear",
            "hanging_fixture"
        ],
    }

    # Dimension tolerance by product type
    DIMENSION_TOLERANCE = {
        "linear": 0.10,      # ±10% for linear fixtures
        "waterproof": 0.10,  # ±10% for waterproof battens
        "panel": 0.0,        # Exact match for panels
        "downlight": 10,     # ±10mm for cutout
        "default": 0.15,     # ±15% default
    }

    def filter(
        self,
        requirement: Requirement,
        products: List[Product],
        *,
        dimension_tolerance_override: Optional[float] = None,
        ip_tolerance_override: Optional[int] = None,
        allow_missing_ip: bool = False
    ) -> FilterResult:
        """
        Filter products against hard constraints.

        Args:
            requirement: The requirement to match
            products: List of all products in catalog

        Returns:
            FilterResult with candidates and eliminated products
        """
        candidates = []
        eliminated = []
        all_checks = []

        for product in products:
            checks = []
            passed = True

            # Check product family compatibility
            family_check = self._check_product_family(requirement, product)
            checks.append(family_check)
            if family_check.status == "FAIL":
                passed = False
                eliminated.append((product, family_check.explanation))
                continue

            # Check dimensions
            dim_check = self._check_dimensions(
                requirement,
                product,
                dimension_tolerance_override=dimension_tolerance_override
            )
            checks.append(dim_check)
            if dim_check.status == "FAIL":
                passed = False
                eliminated.append((product, dim_check.explanation))
                continue

            # Check IP rating
            ip_check = self._check_ip_rating(
                requirement,
                product,
                tolerance_override=ip_tolerance_override,
                allow_missing_ip=allow_missing_ip
            )
            checks.append(ip_check)
            if ip_check.status == "FAIL":
                passed = False
                eliminated.append((product, ip_check.explanation))
                continue

            # Check wattage range
            wattage_check = self._check_wattage(requirement, product)
            checks.append(wattage_check)
            if wattage_check.status == "FAIL":
                passed = False
                eliminated.append((product, wattage_check.explanation))
                continue

            # Check lumens minimum
            lumens_check = self._check_lumens(requirement, product)
            checks.append(lumens_check)
            if lumens_check.status == "FAIL":
                passed = False
                eliminated.append((product, lumens_check.explanation))
                continue

            # Check emergency capability
            emergency_check = self._check_emergency(requirement, product)
            checks.append(emergency_check)
            if emergency_check.status == "FAIL":
                passed = False
                eliminated.append((product, emergency_check.explanation))
                continue

            # Check mounting compatibility
            mounting_check = self._check_mounting(requirement, product)
            checks.append(mounting_check)
            if mounting_check.status == "FAIL":
                passed = False
                eliminated.append((product, mounting_check.explanation))
                continue

            all_checks.extend(checks)
            if passed:
                candidates.append(product)

        return FilterResult(
            candidates=candidates,
            eliminated=eliminated,
            constraint_checks=all_checks,
        )

    def _check_product_family(self, req: Requirement, product: Product) -> ConstraintCheck:
        """Check if product family is compatible with requirement type."""
        req_type = req.requirements.product_type

        if not req_type:
            return ConstraintCheck(
                constraint="product_family",
                status="N/A",
                explanation="No product type specified in requirement",
            )

        compatible_families = self.FAMILY_COMPATIBILITY.get(req_type, [req_type])
        product_family = product.product_family.lower()

        # Check direct match or compatible family
        if product_family in compatible_families or req_type.lower() in product_family:
            return ConstraintCheck(
                constraint="product_family",
                status="PASS",
                explanation=f"Product family '{product_family}' compatible with '{req_type}'",
            )

        return ConstraintCheck(
            constraint="product_family",
            status="FAIL",
            explanation=f"Product family '{product_family}' not compatible with requirement type '{req_type}'",
        )

    def _check_dimensions(
        self,
        req: Requirement,
        product: Product,
        *,
        dimension_tolerance_override: Optional[float] = None
    ) -> ConstraintCheck:
        """Check if product dimensions meet requirements."""
        req_dims = req.requirements.dimensions_mm

        if not req_dims:
            return ConstraintCheck(
                constraint="dimensions",
                status="N/A",
                explanation="No dimension requirement specified",
            )

        product_type = req.requirements.product_type or "default"

        # Parse requirement dimensions
        req_length, req_width = self._parse_dimensions(req_dims)

        if req_length is None:
            return ConstraintCheck(
                constraint="dimensions",
                status="N/A",
                explanation=f"Could not parse dimension requirement: {req_dims}",
            )

        # Get product dimensions
        prod_length = product.length_mm
        prod_width = product.width_mm

        # For panels, check both dimensions
        if product_type == "panel":
            if req_width is not None:
                if prod_length is None or prod_width is None:
                    return ConstraintCheck(
                        constraint="dimensions",
                        status="FAIL",
                        explanation="Panel dimensions missing in product data",
                    )

                if dimension_tolerance_override is not None and dimension_tolerance_override > 0:
                    min_length = req_length * (1 - dimension_tolerance_override)
                    max_length = req_length * (1 + dimension_tolerance_override)
                    min_width = req_width * (1 - dimension_tolerance_override)
                    max_width = req_width * (1 + dimension_tolerance_override)
                    length_match = min_length <= prod_length <= max_length
                    width_match = min_width <= prod_width <= max_width
                    if length_match and width_match:
                        return ConstraintCheck(
                            constraint="dimensions",
                            status="PASS",
                            explanation=(
                                f"Panel dimensions {prod_length}x{prod_width} within "
                                f"tolerance of {req_length}x{req_width} (±{dimension_tolerance_override*100:.0f}%)"
                            ),
                        )
                else:
                    length_match = prod_length == req_length
                    width_match = prod_width == req_width
                    if length_match and width_match:
                        return ConstraintCheck(
                            constraint="dimensions",
                            status="PASS",
                            explanation=f"Panel dimensions {prod_length}x{prod_width} match requirement {req_length}x{req_width}",
                        )

                return ConstraintCheck(
                    constraint="dimensions",
                    status="FAIL",
                    explanation=f"Panel dimensions {prod_length}x{prod_width} don't match requirement {req_length}x{req_width}",
                )

        # For linear fixtures, check length with tolerance
        if product_type in ["linear", "waterproof"]:
            tolerance = self.DIMENSION_TOLERANCE.get(product_type, 0.10)
            if dimension_tolerance_override is not None:
                tolerance = dimension_tolerance_override

            if prod_length is None:
                return ConstraintCheck(
                    constraint="dimensions",
                    status="FAIL",
                    explanation=f"Product has no length specification for {product_type} fixture",
                )

            min_length = req_length * (1 - tolerance)
            max_length = req_length * (1 + tolerance)

            if min_length <= prod_length <= max_length:
                return ConstraintCheck(
                    constraint="dimensions",
                    status="PASS",
                    explanation=f"Product length {prod_length}mm within tolerance of {req_length}mm (±{tolerance*100:.0f}%)",
                )
            else:
                return ConstraintCheck(
                    constraint="dimensions",
                    status="FAIL",
                    explanation=f"Product length {prod_length}mm outside tolerance of {req_length}mm (±{tolerance*100:.0f}%)",
                )

        return ConstraintCheck(
            constraint="dimensions",
            status="N/A",
            explanation="Dimension check not critical for this product type",
        )

    def _parse_dimensions(self, dims_str: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse dimension string into length and width."""
        if not dims_str:
            return None, None

        # Handle "1200", "1200mm", "600x600", "600*600", "600 x 600 mm", "24\"x24\""
        dims_str = dims_str.lower().strip()

        # Normalize separators
        cleaned = dims_str.replace("×", "x").replace("*", "x")
        cleaned = cleaned.replace("mm", " mm").replace("in", " in").replace("\"", " in")
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Extract numeric values with optional unit
        parts = [p.strip() for p in cleaned.split("x")]
        if len(parts) == 1:
            value = self._parse_dimension_value(parts[0])
            return value, None

        if len(parts) >= 2:
            length = self._parse_dimension_value(parts[0])
            width = self._parse_dimension_value(parts[1])
            return length, width

        return None, None

    def _parse_dimension_value(self, raw: str) -> Optional[int]:
        """Parse a single dimension value, converting inches to mm when needed."""
        if not raw:
            return None

        match = re.search(r"([\d.]+)\s*(mm|in)?", raw)
        if not match:
            return None

        value = float(match.group(1))
        unit = match.group(2)
        if unit == "in":
            value *= 25.4
        return int(round(value))

    def _check_ip_rating(
        self,
        req: Requirement,
        product: Product,
        *,
        tolerance_override: Optional[int] = None,
        allow_missing_ip: bool = False
    ) -> ConstraintCheck:
        """Check if product IP rating meets minimum requirement."""
        req_ip = req.requirements.min_ip_rating

        if req_ip is None:
            return ConstraintCheck(
                constraint="ip_rating",
                status="N/A",
                explanation="No IP rating requirement specified",
            )

        prod_ip = product.ip_rating

        if prod_ip is None:
            if allow_missing_ip:
                return ConstraintCheck(
                    constraint="ip_rating",
                    status="N/A",
                    explanation=f"Product has no IP rating; requirement is IP{req_ip}",
                )
            return ConstraintCheck(
                constraint="ip_rating",
                status="FAIL",
                explanation=f"Product has no IP rating but IP{req_ip} required",
            )

        # Allow 1-point tolerance for high IP ratings
        tolerance = 1 if req_ip >= 65 else 0
        if tolerance_override is not None:
            tolerance = max(tolerance_override, 0)
        min_acceptable = req_ip - tolerance

        if prod_ip >= min_acceptable:
            return ConstraintCheck(
                constraint="ip_rating",
                status="PASS",
                explanation=f"Product IP{prod_ip} meets requirement IP{req_ip}",
            )

        return ConstraintCheck(
            constraint="ip_rating",
            status="FAIL",
            explanation=f"Product IP{prod_ip} does not meet requirement IP{req_ip}",
        )

    def _check_wattage(self, req: Requirement, product: Product) -> ConstraintCheck:
        """Check if product wattage is within acceptable range."""
        wattage_spec = req.requirements.wattage

        if wattage_spec is None or wattage_spec.target is None:
            return ConstraintCheck(
                constraint="wattage",
                status="N/A",
                explanation="No wattage requirement specified",
            )

        prod_wattage = product.wattage

        if prod_wattage is None:
            return ConstraintCheck(
                constraint="wattage",
                status="N/A",
                explanation="Product wattage not specified",
            )

        # Use specified range or default ±20%
        min_w = wattage_spec.min if wattage_spec.min else wattage_spec.target * 0.8
        max_w = wattage_spec.max if wattage_spec.max else wattage_spec.target * 1.2

        if min_w <= prod_wattage <= max_w:
            return ConstraintCheck(
                constraint="wattage",
                status="PASS",
                explanation=f"Product wattage {prod_wattage}W within range {min_w}-{max_w}W",
            )

        return ConstraintCheck(
            constraint="wattage",
            status="FAIL",
            explanation=f"Product wattage {prod_wattage}W outside range {min_w}-{max_w}W",
        )

    def _check_lumens(self, req: Requirement, product: Product) -> ConstraintCheck:
        """Check if product lumens meets minimum requirement."""
        lumens_spec = req.requirements.lumens

        if lumens_spec is None:
            return ConstraintCheck(
                constraint="lumens",
                status="N/A",
                explanation="No lumens requirement specified",
            )

        prod_lumens = product.lumens

        if prod_lumens is None:
            # Some products (like exit signs, strips) may not have lumens
            return ConstraintCheck(
                constraint="lumens",
                status="N/A",
                explanation="Product lumens not specified",
            )

        # Minimum is a hard constraint
        min_lumens = lumens_spec.min if lumens_spec.min else (lumens_spec.target * 0.9 if lumens_spec.target else 0)

        if prod_lumens >= min_lumens:
            # Check if excessively over
            if lumens_spec.target and prod_lumens > lumens_spec.target * 1.5:
                return ConstraintCheck(
                    constraint="lumens",
                    status="PASS",
                    explanation=f"Product lumens {prod_lumens}lm exceeds requirement significantly (target {lumens_spec.target}lm)",
                )
            return ConstraintCheck(
                constraint="lumens",
                status="PASS",
                explanation=f"Product lumens {prod_lumens}lm meets minimum {min_lumens}lm",
            )

        return ConstraintCheck(
            constraint="lumens",
            status="FAIL",
            explanation=f"Product lumens {prod_lumens}lm below minimum {min_lumens}lm",
        )

    def _check_emergency(self, req: Requirement, product: Product) -> ConstraintCheck:
        """Check if product supports required emergency type."""
        emergency_type = req.requirements.emergency_type

        if not emergency_type:
            return ConstraintCheck(
                constraint="emergency",
                status="N/A",
                explanation="No emergency requirement specified",
            )

        # Central generator - standard products work fine
        if emergency_type == "central_generator":
            return ConstraintCheck(
                constraint="emergency",
                status="PASS",
                explanation="Standard product compatible with central generator backup",
            )

        # Battery pack or central battery - need specific support
        if product.supports_emergency(emergency_type):
            duration_req = req.requirements.emergency_duration_hours
            if duration_req and product.emergency_duration:
                if product.emergency_duration >= duration_req:
                    return ConstraintCheck(
                        constraint="emergency",
                        status="PASS",
                        explanation=f"Product supports {emergency_type} with {product.emergency_duration}h duration",
                    )
                else:
                    return ConstraintCheck(
                        constraint="emergency",
                        status="FAIL",
                        explanation=f"Product emergency duration {product.emergency_duration}h < required {duration_req}h",
                    )
            return ConstraintCheck(
                constraint="emergency",
                status="PASS",
                explanation=f"Product supports {emergency_type}",
            )

        return ConstraintCheck(
            constraint="emergency",
            status="FAIL",
            explanation=f"Product does not support {emergency_type} emergency type",
        )

    def _check_mounting(self, req: Requirement, product: Product) -> ConstraintCheck:
        """Check if product supports required mounting type."""
        req_mounting = req.requirements.mounting

        if not req_mounting:
            return ConstraintCheck(
                constraint="mounting",
                status="N/A",
                explanation="No mounting requirement specified",
            )

        if not product.mounting_options:
            return ConstraintCheck(
                constraint="mounting",
                status="N/A",
                explanation="Product mounting options not specified",
            )

        # Normalize mounting name
        req_mounting_lower = req_mounting.lower()
        product_mountings = [m.lower() for m in product.mounting_options]

        if req_mounting_lower in product_mountings:
            return ConstraintCheck(
                constraint="mounting",
                status="PASS",
                explanation=f"Product supports {req_mounting} mounting",
            )

        # Check for equivalent mounting types
        mounting_equivalents = {
            "ceiling": ["surface", "recessed", "suspended"],
            "surface": ["ceiling"],
            "recessed": ["ceiling"],
        }

        equivalents = mounting_equivalents.get(req_mounting_lower, [])
        for equiv in equivalents:
            if equiv in product_mountings:
                return ConstraintCheck(
                    constraint="mounting",
                    status="PASS",
                    explanation=f"Product {equiv} mounting compatible with {req_mounting}",
                )

        return ConstraintCheck(
            constraint="mounting",
            status="FAIL",
            explanation=f"Product does not support {req_mounting} mounting (has: {', '.join(product.mounting_options)})",
        )
