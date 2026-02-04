"""
Tests for the lighting product matching system.

Run with: python -m pytest tests/ -v
"""

import json
import pytest
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.requirement import Requirement, RequirementSpecs, WattageSpec, LumensSpec
from src.models.product import Product
from src.models.match_result import ConfidenceLevel
from src.validation.source_validator import SourceValidator, ValidationSeverity
from src.matching.constraint_filter import ConstraintFilter
from src.matching.preference_scorer import PreferenceScorer
from src.matching.confidence_classifier import ConfidenceClassifier
from src.matching.matching_engine import MatchingEngine, MatchingConfig
from src.utils.variant_resolver import VariantResolver


class TestSourceValidator:
    """Tests for source validation."""

    def test_high_confidence_passes(self):
        """High confidence requirements should pass validation."""
        req = Requirement(
            line_item="L1",
            quantity=100,
            unit="nr",
            description="Test fixture",
            requirements=RequirementSpecs(
                product_type="waterproof",
                wattage=WattageSpec(target=30),
                dimensions_mm="1200",
            ),
            confidence=0.95,
        )

        validator = SourceValidator()
        result = validator.validate(req)

        assert result.is_valid
        assert result.can_match

    def test_low_confidence_fails(self):
        """Very low confidence requirements should fail."""
        req = Requirement(
            line_item="X1",
            quantity=10,
            unit="nr",
            description="Unknown",
            requirements=RequirementSpecs(product_type="downlight"),
            confidence=0.3,
        )

        validator = SourceValidator()
        result = validator.validate(req)

        assert not result.can_match

    def test_missing_critical_specs_detected(self):
        """Missing critical specs should be flagged."""
        req = Requirement(
            line_item="P1",
            quantity=50,
            unit="nr",
            description="Panel",
            requirements=RequirementSpecs(
                product_type="panel",
                # Missing dimensions_mm which is critical for panels
                wattage=WattageSpec(target=30),
            ),
            confidence=0.9,
        )

        validator = SourceValidator()
        result = validator.validate(req)

        # Should have critical issue for missing dimensions
        critical_issues = [i for i in result.issues if i.severity == ValidationSeverity.CRITICAL]
        assert len(critical_issues) > 0


class TestConstraintFilter:
    """Tests for hard constraint filtering."""

    @pytest.fixture
    def sample_products(self):
        """Create sample products for testing."""
        return [
            Product(
                sku="WPB-1200-30",
                name="Waterproof Batten 1200mm",
                description="30W waterproof batten",
                product_family="waterproof_batten",
                mounting_options=["surface", "wall"],
                wattage=30,
                lumens=3800,
                ip_rating=66,
                length_mm=1200,
            ),
            Product(
                sku="WPB-1500-36",
                name="Waterproof Batten 1500mm",
                description="36W waterproof batten",
                product_family="waterproof_batten",
                mounting_options=["surface"],
                wattage=36,
                lumens=4500,
                ip_rating=66,
                length_mm=1500,
            ),
            Product(
                sku="PNL-600x600-30",
                name="Panel 600x600",
                description="30W panel",
                product_family="panel",
                mounting_options=["recessed"],
                wattage=30,
                lumens=4000,
                ip_rating=20,
                length_mm=600,
                width_mm=600,
            ),
        ]

    def test_dimension_filtering(self, sample_products):
        """Products outside dimension tolerance should be eliminated."""
        req = Requirement(
            line_item="L1",
            quantity=100,
            unit="nr",
            description="1200mm batten",
            requirements=RequirementSpecs(
                product_type="waterproof",
                dimensions_mm="1200",
            ),
            confidence=0.95,
        )

        filter = ConstraintFilter()
        result = filter.filter(req, sample_products)

        # Only 1200mm batten should pass
        assert len(result.candidates) == 1
        assert result.candidates[0].sku == "WPB-1200-30"

    def test_ip_rating_filtering(self, sample_products):
        """Products below IP rating should be eliminated."""
        req = Requirement(
            line_item="WET1",
            quantity=10,
            unit="nr",
            description="Wet area fixture",
            requirements=RequirementSpecs(
                product_type="waterproof",
                min_ip_rating=65,
                dimensions_mm="1200",
            ),
            confidence=0.95,
        )

        filter = ConstraintFilter()
        result = filter.filter(req, sample_products)

        # IP66 products should pass for IP65 requirement
        assert len(result.candidates) == 1
        assert result.candidates[0].ip_rating >= 65

    def test_product_family_filtering(self, sample_products):
        """Incompatible product families should be eliminated."""
        req = Requirement(
            line_item="P1",
            quantity=50,
            unit="nr",
            description="Panel",
            requirements=RequirementSpecs(
                product_type="panel",
                dimensions_mm="600x600",
            ),
            confidence=0.95,
        )

        filter = ConstraintFilter()
        result = filter.filter(req, sample_products)

        # Only panel should match
        assert len(result.candidates) == 1
        assert result.candidates[0].product_family == "panel"


class TestPreferenceScorer:
    """Tests for soft preference scoring."""

    def test_exact_match_scores_highest(self):
        """Products with exact spec matches should score highest."""
        products = [
            Product(
                sku="EXACT",
                name="Exact Match",
                description="",
                product_family="downlight",
                wattage=22,
                lumens=2100,
                color_temp_k=3000,
            ),
            Product(
                sku="CLOSE",
                name="Close Match",
                description="",
                product_family="downlight",
                wattage=25,
                lumens=2000,
                color_temp_k=4000,
            ),
        ]

        req = Requirement(
            line_item="D1",
            quantity=100,
            unit="nr",
            description="Downlight",
            requirements=RequirementSpecs(
                product_type="downlight",
                wattage=WattageSpec(target=22),
                lumens=LumensSpec(target=2100),
            ),
            confidence=0.95,
        )

        scorer = PreferenceScorer()
        scored = scorer.score_candidates(req, products)

        # Exact match should be first
        assert scored[0].product.sku == "EXACT"
        assert scored[0].score > scored[1].score


class TestConfidenceClassifier:
    """Tests for confidence classification."""

    def test_high_confidence_auto_approves(self):
        """High confidence matches should be auto-approved."""
        classifier = ConfidenceClassifier()

        level, needs_review, reason = classifier.classify(
            confidence_score=0.95,
            candidates_count=1,
            selection_method="single_candidate",
        )

        assert level == ConfidenceLevel.AUTO_APPROVE
        assert not needs_review

    def test_low_confidence_requires_review(self):
        """Low confidence matches should require review."""
        classifier = ConfidenceClassifier()

        level, needs_review, reason = classifier.classify(
            confidence_score=0.55,
            candidates_count=5,
            has_warnings=True,
        )

        assert level == ConfidenceLevel.REVIEW_REQUIRED
        assert needs_review

    def test_no_candidates_returns_no_match(self):
        """Zero candidates should return NO_MATCH."""
        classifier = ConfidenceClassifier()

        level, needs_review, reason = classifier.classify(
            confidence_score=0.0,
            candidates_count=0,
        )

        assert level == ConfidenceLevel.NO_MATCH
        assert needs_review


class TestVariantResolver:
    """Tests for variant resolution."""

    def test_identifies_variants(self):
        """Should correctly identify base/variant relationships."""
        requirements = [
            Requirement(
                line_item="L1",
                quantity=100,
                unit="nr",
                description="Base",
                requirements=RequirementSpecs(
                    product_type="waterproof",
                    dimensions_mm="1200",
                    wattage=WattageSpec(target=30),
                ),
                confidence=0.95,
            ),
            Requirement(
                line_item="L1/B",
                quantity=50,
                unit="nr",
                description="Battery variant",
                requirements=RequirementSpecs(
                    product_type="waterproof",
                    emergency_type="battery_pack",
                    # Missing dimensions and wattage
                ),
                confidence=0.85,
            ),
            Requirement(
                line_item="L1/E",
                quantity=30,
                unit="nr",
                description="Generator variant",
                requirements=RequirementSpecs(
                    product_type="waterproof",
                    emergency_type="central_generator",
                    # Missing dimensions and wattage
                ),
                confidence=0.85,
            ),
        ]

        resolver = VariantResolver()
        relationships = resolver.get_variant_relationships(requirements)

        assert "L1" in relationships
        assert "L1/B" in relationships["L1"]
        assert "L1/E" in relationships["L1"]

    def test_inherits_missing_specs(self):
        """Variants should inherit missing specs from base."""
        requirements = [
            Requirement(
                line_item="D1",
                quantity=100,
                unit="nr",
                description="Base downlight",
                requirements=RequirementSpecs(
                    product_type="downlight",
                    wattage=WattageSpec(target=22),
                    lumens=LumensSpec(target=2100),
                ),
                confidence=0.95,
            ),
            Requirement(
                line_item="D1/B",
                quantity=50,
                unit="nr",
                description="Battery variant",
                requirements=RequirementSpecs(
                    product_type="downlight",
                    emergency_type="battery_pack",
                    # Missing wattage and lumens
                ),
                confidence=0.85,
            ),
        ]

        resolver = VariantResolver()
        resolved = resolver.resolve_variants(requirements)

        # Find the D1/B variant
        variant = next(r for r in resolved if r.line_item == "D1/B")

        # Should have inherited wattage and lumens
        assert variant.requirements.wattage is not None
        assert variant.requirements.wattage.target == 22
        assert variant.requirements.lumens is not None
        assert variant.requirements.lumens.target == 2100


class TestMatchingEngine:
    """Integration tests for the full matching engine."""

    @pytest.fixture
    def engine(self):
        """Create engine with test catalog."""
        config = MatchingConfig(skip_validation=False)
        engine = MatchingEngine(config)

        # Load the actual catalog
        catalog_path = Path(__file__).parent.parent / "data" / "product_catalog.json"
        if catalog_path.exists():
            engine.load_catalog(str(catalog_path))
        else:
            # Create minimal test catalog
            engine.products = [
                Product(
                    sku="TEST-WPB-1200-30",
                    name="Test Waterproof Batten",
                    description="Test product",
                    product_family="waterproof_batten",
                    mounting_options=["surface", "wall"],
                    wattage=30,
                    lumens=3800,
                    ip_rating=66,
                    length_mm=1200,
                    color_temp_k=3000,
                    emergency_options=["standard", "battery_pack"],
                    emergency_duration=3,
                ),
            ]

        return engine

    def test_full_matching_pipeline(self, engine):
        """Test complete matching from requirement to result."""
        req = Requirement(
            line_item="TEST-L1",
            quantity=10,
            unit="nr",
            description="Test waterproof batten",
            requirements=RequirementSpecs(
                product_type="waterproof",
                mounting="surface",
                min_ip_rating=66,
                wattage=WattageSpec(target=30, min=27, max=33),
                lumens=LumensSpec(target=3800, min=3400, max=4200),
                dimensions_mm="1200",
            ),
            confidence=0.95,
        )

        result = engine.match_single(req)

        # Should find a match
        assert result.matched_sku is not None
        assert result.confidence > 0.5

    def test_no_match_scenario(self, engine):
        """Test when no products can match."""
        req = Requirement(
            line_item="IMPOSSIBLE",
            quantity=1,
            unit="nr",
            description="Impossible requirements",
            requirements=RequirementSpecs(
                product_type="waterproof",
                min_ip_rating=68,  # Higher than any product
                wattage=WattageSpec(target=1000),  # Way higher than available
                dimensions_mm="9999",  # No such size
            ),
            confidence=0.95,
        )

        result = engine.match_single(req)

        assert result.confidence_level == ConfidenceLevel.NO_MATCH
        assert result.matched_sku is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
