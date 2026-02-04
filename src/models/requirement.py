"""Data models for lighting requirements."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class WattageSpec:
    """Wattage specification with target and acceptable range."""
    target: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["WattageSpec"]:
        if data is None:
            return None
        return cls(
            target=data.get("target"),
            min=data.get("min"),
            max=data.get("max")
        )


@dataclass
class LumensSpec:
    """Lumens specification with target and acceptable range."""
    target: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["LumensSpec"]:
        if data is None:
            return None
        return cls(
            target=data.get("target"),
            min=data.get("min"),
            max=data.get("max")
        )


@dataclass
class ColorTempSpec:
    """Color temperature specification."""
    target: Optional[int] = None
    acceptable: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["ColorTempSpec"]:
        if data is None:
            return None
        return cls(
            target=data.get("target"),
            acceptable=data.get("acceptable", [])
        )


@dataclass
class RequirementSpecs:
    """Full specification requirements for a lighting fixture."""
    product_type: Optional[str] = None
    mounting: Optional[str] = None
    environment: Optional[str] = None
    min_ip_rating: Optional[int] = None

    wattage: Optional[WattageSpec] = None
    lumens: Optional[LumensSpec] = None
    color_temp_k: Optional[ColorTempSpec] = None

    dimensions_mm: Optional[str] = None
    beam_angle: Optional[Dict] = None
    cutout_mm: Optional[Dict] = None

    dimmable: Optional[bool] = None
    dimming_type: Optional[str] = None

    emergency_type: Optional[str] = None
    emergency_duration_hours: Optional[int] = None

    reference_manufacturer: Optional[str] = None
    reference_catalog: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "RequirementSpecs":
        return cls(
            product_type=data.get("product_type"),
            mounting=data.get("mounting"),
            environment=data.get("environment"),
            min_ip_rating=data.get("min_ip_rating"),
            wattage=WattageSpec.from_dict(data.get("wattage")),
            lumens=LumensSpec.from_dict(data.get("lumens")),
            color_temp_k=ColorTempSpec.from_dict(data.get("color_temp_k")),
            dimensions_mm=data.get("dimensions_mm"),
            beam_angle=data.get("beam_angle"),
            cutout_mm=data.get("cutout_mm"),
            dimmable=data.get("dimmable"),
            dimming_type=data.get("dimming_type"),
            emergency_type=data.get("emergency_type"),
            emergency_duration_hours=data.get("emergency_duration_hours"),
            reference_manufacturer=data.get("reference_manufacturer"),
            reference_catalog=data.get("reference_catalog"),
        )


@dataclass
class Requirement:
    """A single lighting requirement from the customer."""
    line_item: str
    quantity: int
    unit: str
    description: str
    requirements: RequirementSpecs
    source_docs: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict) -> "Requirement":
        return cls(
            line_item=data.get("line_item", ""),
            quantity=data.get("quantity", 0),
            unit=data.get("unit", "nr"),
            description=data.get("description", ""),
            requirements=RequirementSpecs.from_dict(data.get("requirements", {})),
            source_docs=data.get("source_docs", []),
            notes=data.get("notes", []),
            confidence=data.get("confidence", 1.0),
        )

    def has_performance_spec(self) -> bool:
        """Check if requirement has at least one performance specification."""
        return any([
            self.requirements.wattage is not None and self.requirements.wattage.target is not None,
            self.requirements.lumens is not None and self.requirements.lumens.target is not None,
            self.requirements.reference_catalog is not None,
        ])

    def has_dimension_spec(self) -> bool:
        """Check if requirement has dimension specification."""
        return self.requirements.dimensions_mm is not None

    def is_linear_or_panel(self) -> bool:
        """Check if product type requires dimension specification."""
        return self.requirements.product_type in ["waterproof", "linear", "panel"]
