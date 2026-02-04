"""Soft preference scoring for product matching.

Scores products that pass hard constraints based on how well
they match soft preferences like target wattage, lumens, color temp, etc.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math

from ..models.requirement import Requirement
from ..models.product import Product


@dataclass
class ScoreBreakdown:
    """Breakdown of scoring components."""
    wattage_score: float = 0.0
    lumens_score: float = 0.0
    color_temp_score: float = 0.0
    reference_score: float = 0.0
    feature_score: float = 0.0
    efficacy_bonus: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "wattage_score": self.wattage_score,
            "lumens_score": self.lumens_score,
            "color_temp_score": self.color_temp_score,
            "reference_score": self.reference_score,
            "feature_score": self.feature_score,
            "efficacy_bonus": self.efficacy_bonus,
        }


@dataclass
class ScoredCandidate:
    """A candidate product with its preference score."""
    product: Product
    score: float
    breakdown: ScoreBreakdown
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "sku": self.product.sku,
            "name": self.product.name,
            "score": self.score,
            "breakdown": self.breakdown.to_dict(),
            "notes": self.notes,
        }


class PreferenceScorer:
    """Scores products based on soft preference matching."""

    # Scoring weights (should sum to ~1.0 for base score)
    WEIGHTS = {
        "wattage": 0.20,
        "lumens": 0.25,
        "color_temp": 0.15,
        "reference": 0.25,
        "features": 0.15,
    }

    # Efficacy bonus (added on top of base score)
    EFFICACY_GOOD_THRESHOLD = 100  # lm/W
    EFFICACY_EXCELLENT_THRESHOLD = 120  # lm/W
    EFFICACY_BONUS_GOOD = 0.02
    EFFICACY_BONUS_EXCELLENT = 0.05

    def score_candidates(
        self,
        requirement: Requirement,
        candidates: List[Product]
    ) -> List[ScoredCandidate]:
        """
        Score all candidate products against requirement preferences.

        Args:
            requirement: The requirement to match
            candidates: Products that passed hard constraint filtering

        Returns:
            List of ScoredCandidate sorted by score (highest first)
        """
        scored = []

        for product in candidates:
            breakdown = ScoreBreakdown()
            notes = []

            # Score each preference category
            breakdown.wattage_score = self._score_wattage(requirement, product, notes)
            breakdown.lumens_score = self._score_lumens(requirement, product, notes)
            breakdown.color_temp_score = self._score_color_temp(requirement, product, notes)
            breakdown.reference_score = self._score_reference(requirement, product, notes)
            breakdown.feature_score = self._score_features(requirement, product, notes)
            breakdown.efficacy_bonus = self._score_efficacy(product, notes)

            # Calculate weighted score
            base_score = (
                breakdown.wattage_score * self.WEIGHTS["wattage"] +
                breakdown.lumens_score * self.WEIGHTS["lumens"] +
                breakdown.color_temp_score * self.WEIGHTS["color_temp"] +
                breakdown.reference_score * self.WEIGHTS["reference"] +
                breakdown.feature_score * self.WEIGHTS["features"]
            )

            # Add efficacy bonus
            total_score = min(1.0, base_score + breakdown.efficacy_bonus)

            scored.append(ScoredCandidate(
                product=product,
                score=total_score,
                breakdown=breakdown,
                notes=notes,
            ))

        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)

        return scored

    def _score_wattage(
        self,
        req: Requirement,
        product: Product,
        notes: List[str]
    ) -> float:
        """Score based on how close product wattage is to target."""
        wattage_spec = req.requirements.wattage

        if wattage_spec is None or wattage_spec.target is None:
            return 0.5  # Neutral score if no preference

        if product.wattage is None:
            return 0.3  # Low score if product has no wattage

        target = wattage_spec.target
        actual = product.wattage

        # Calculate percentage difference
        diff_pct = abs(actual - target) / target

        if diff_pct <= 0.05:
            return 1.0  # Within 5% is perfect
        elif diff_pct <= 0.10:
            return 0.9  # Within 10% is excellent
        elif diff_pct <= 0.20:
            return 0.7  # Within 20% is good
        elif diff_pct <= 0.30:
            notes.append(f"Wattage {actual}W differs from target {target}W by {diff_pct*100:.0f}%")
            return 0.5  # Within 30% is acceptable
        else:
            notes.append(f"Wattage {actual}W significantly differs from target {target}W")
            return 0.3  # More than 30% is poor match

    def _score_lumens(
        self,
        req: Requirement,
        product: Product,
        notes: List[str]
    ) -> float:
        """Score based on how close product lumens is to target."""
        lumens_spec = req.requirements.lumens

        if lumens_spec is None or lumens_spec.target is None:
            return 0.5  # Neutral score if no preference

        if product.lumens is None:
            return 0.3  # Low score if product has no lumens

        target = lumens_spec.target
        actual = product.lumens

        # Calculate percentage difference
        diff_pct = abs(actual - target) / target

        if diff_pct <= 0.05:
            return 1.0  # Within 5% is perfect
        elif diff_pct <= 0.10:
            return 0.95  # Within 10% is excellent
        elif diff_pct <= 0.15:
            return 0.85  # Within 15% is very good
        elif diff_pct <= 0.20:
            return 0.7  # Within 20% is good
        else:
            # Check if over or under
            if actual > target:
                # Over is usually acceptable
                if actual <= target * 1.5:
                    return 0.6
                else:
                    notes.append(f"Lumens {actual}lm significantly exceeds target {target}lm")
                    return 0.4
            else:
                notes.append(f"Lumens {actual}lm below target {target}lm by {diff_pct*100:.0f}%")
                return 0.4

    def _score_color_temp(
        self,
        req: Requirement,
        product: Product,
        notes: List[str]
    ) -> float:
        """Score based on color temperature match."""
        color_spec = req.requirements.color_temp_k

        if color_spec is None or color_spec.target is None:
            return 0.5  # Neutral score if no preference

        product_temps = product.get_effective_color_temp()

        if not product_temps:
            # Check if product is tunable (no fixed temp)
            if product.color_temp_k is None and product.color_temp_options:
                notes.append("Product has tunable color temperature")
                return 0.9  # Tunable products are usually good
            return 0.4  # No color temp info

        target = color_spec.target
        acceptable = color_spec.acceptable if color_spec.acceptable else [target]

        # Check for exact match with acceptable values
        for temp in product_temps:
            if temp in acceptable:
                return 1.0  # Exact match

        # Check for exact match with target
        if target in product_temps:
            return 1.0

        # Find closest match
        closest = min(product_temps, key=lambda x: abs(x - target))
        diff = abs(closest - target)

        if diff <= 300:
            return 0.9  # Very close
        elif diff <= 500:
            return 0.7  # Close
        elif diff <= 1000:
            notes.append(f"Color temp {closest}K differs from target {target}K")
            return 0.5  # Moderate difference
        else:
            notes.append(f"Color temp {closest}K significantly differs from target {target}K")
            return 0.3  # Large difference

    def _score_reference(
        self,
        req: Requirement,
        product: Product,
        notes: List[str]
    ) -> float:
        """Score based on reference product/catalog similarity."""
        ref_catalog = req.requirements.reference_catalog
        ref_mfr = req.requirements.reference_manufacturer

        if not ref_catalog and not ref_mfr:
            return 0.5  # Neutral score if no reference

        score = 0.5  # Start with neutral

        # Check if product is compatible with reference
        if ref_catalog:
            # Check compatible_with and replaces lists
            for compat in product.compatible_with:
                if self._catalog_similarity(ref_catalog, compat) > 0.5:
                    score = max(score, 0.95)
                    notes.append(f"Product compatible with reference {compat}")
                    break

            for replacement in product.replaces:
                if self._catalog_similarity(ref_catalog, replacement) > 0.5:
                    score = max(score, 1.0)
                    notes.append(f"Product replaces reference {replacement}")
                    break

            # Check SKU/name similarity to catalog number
            sku_similarity = self._catalog_similarity(ref_catalog, product.sku)
            name_similarity = self._catalog_similarity(ref_catalog, product.name)

            if sku_similarity > 0.3 or name_similarity > 0.3:
                score = max(score, 0.7 + max(sku_similarity, name_similarity) * 0.3)

        return score

    def _catalog_similarity(self, ref: str, product_str: str) -> float:
        """Calculate similarity between reference catalog and product string."""
        if not ref or not product_str:
            return 0.0

        ref_lower = ref.lower()
        prod_lower = product_str.lower()

        # Check for common patterns
        ref_parts = set(ref_lower.replace("-", " ").replace("/", " ").split())
        prod_parts = set(prod_lower.replace("-", " ").replace("/", " ").split())

        if not ref_parts or not prod_parts:
            return 0.0

        common = ref_parts & prod_parts
        similarity = len(common) / max(len(ref_parts), len(prod_parts))

        return similarity

    def _score_features(
        self,
        req: Requirement,
        product: Product,
        notes: List[str]
    ) -> float:
        """Score based on feature alignment (dimming, etc.)."""
        score = 0.5  # Start neutral
        feature_count = 0
        feature_matches = 0

        # Check dimming
        if req.requirements.dimmable is not None:
            feature_count += 1
            if req.requirements.dimmable == product.dimmable:
                feature_matches += 1
            elif req.requirements.dimmable and not product.dimmable:
                notes.append("Dimming required but product not dimmable")

        # Check dimming type
        if req.requirements.dimming_type:
            feature_count += 1
            if req.requirements.dimming_type.upper() in [d.upper() for d in product.dimming_types]:
                feature_matches += 1
            elif product.dimming_types:
                notes.append(f"Dimming type {req.requirements.dimming_type} not supported (has: {', '.join(product.dimming_types)})")

        # Calculate feature score
        if feature_count > 0:
            score = feature_matches / feature_count
        else:
            # No specific features required, give good score if product has extras
            if product.dimmable:
                score += 0.1
            if product.dimming_types:
                score += 0.1
            score = min(1.0, score)

        return score

    def _score_efficacy(self, product: Product, notes: List[str]) -> float:
        """Calculate efficacy bonus."""
        efficacy = product.efficacy

        if efficacy is None and product.wattage and product.lumens:
            efficacy = product.lumens / product.wattage

        if efficacy is None:
            return 0.0

        if efficacy >= self.EFFICACY_EXCELLENT_THRESHOLD:
            notes.append(f"Excellent efficacy: {efficacy:.0f} lm/W")
            return self.EFFICACY_BONUS_EXCELLENT
        elif efficacy >= self.EFFICACY_GOOD_THRESHOLD:
            return self.EFFICACY_BONUS_GOOD
        elif efficacy < 80:
            notes.append(f"Low efficacy: {efficacy:.0f} lm/W")

        return 0.0
