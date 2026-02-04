"""Data model for products in the catalog."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class Product:
    """A product from the lighting catalog."""
    sku: str
    name: str
    description: str

    # Product classification
    product_family: str
    mounting_options: List[str] = field(default_factory=list)

    # Performance specs
    wattage: Optional[float] = None
    lumens: Optional[float] = None
    efficacy: Optional[float] = None
    color_temp_k: Optional[int] = None
    color_temp_options: List[int] = field(default_factory=list)
    beam_angle: Optional[int] = None
    cri: Optional[int] = None

    # Protection ratings
    ip_rating: Optional[int] = None
    ik_rating: Optional[int] = None

    # Dimensions
    length_mm: Optional[int] = None
    width_mm: Optional[int] = None
    height_mm: Optional[int] = None
    cutout_mm: Optional[int] = None
    dimensions_mm: Optional[str] = None

    # Control options
    dimmable: bool = False
    dimming_types: List[str] = field(default_factory=list)

    # Emergency options
    emergency_options: List[str] = field(default_factory=list)
    emergency_duration: Optional[int] = None

    # Additional info
    certifications: List[str] = field(default_factory=list)
    warranty_years: Optional[int] = None
    compatible_with: List[str] = field(default_factory=list)
    replaces: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> "Product":
        return cls(
            sku=data.get("sku", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            product_family=data.get("product_family", ""),
            mounting_options=data.get("mounting_options", []),
            wattage=data.get("wattage"),
            lumens=data.get("lumens"),
            efficacy=data.get("efficacy"),
            color_temp_k=data.get("color_temp_k"),
            color_temp_options=data.get("color_temp_options", []),
            beam_angle=data.get("beam_angle"),
            cri=data.get("cri"),
            ip_rating=data.get("ip_rating"),
            ik_rating=data.get("ik_rating"),
            length_mm=data.get("length_mm"),
            width_mm=data.get("width_mm"),
            height_mm=data.get("height_mm"),
            cutout_mm=data.get("cutout_mm"),
            dimensions_mm=data.get("dimensions_mm"),
            dimmable=data.get("dimmable", False),
            dimming_types=data.get("dimming_types", []),
            emergency_options=data.get("emergency_options", []),
            emergency_duration=data.get("emergency_duration"),
            certifications=data.get("certifications", []),
            warranty_years=data.get("warranty_years"),
            compatible_with=data.get("compatible_with", []),
            replaces=data.get("replaces", []),
        )

    def to_dict(self) -> Dict:
        return {
            "sku": self.sku,
            "name": self.name,
            "description": self.description,
            "product_family": self.product_family,
            "mounting_options": self.mounting_options,
            "wattage": self.wattage,
            "lumens": self.lumens,
            "efficacy": self.efficacy,
            "color_temp_k": self.color_temp_k,
            "color_temp_options": self.color_temp_options,
            "beam_angle": self.beam_angle,
            "cri": self.cri,
            "ip_rating": self.ip_rating,
            "ik_rating": self.ik_rating,
            "length_mm": self.length_mm,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "cutout_mm": self.cutout_mm,
            "dimensions_mm": self.dimensions_mm,
            "dimmable": self.dimmable,
            "dimming_types": self.dimming_types,
            "emergency_options": self.emergency_options,
            "emergency_duration": self.emergency_duration,
            "certifications": self.certifications,
            "warranty_years": self.warranty_years,
            "compatible_with": self.compatible_with,
            "replaces": self.replaces,
        }

    def supports_emergency(self, emergency_type: str) -> bool:
        """Check if product supports a specific emergency type."""
        if not self.emergency_options:
            return emergency_type == "central_generator"  # Standard products work with generators

        type_mapping = {
            "battery_pack": ["battery", "battery_pack", "self_contained", "em", "emergency"],
            "central_generator": ["standard", "generator", "central"],
            "central_battery": ["standard", "central", "central_battery"],
        }

        check_terms = type_mapping.get(emergency_type, [emergency_type])
        return any(
            term.lower() in opt.lower()
            for opt in self.emergency_options
            for term in check_terms
        )

    def get_effective_color_temp(self) -> List[int]:
        """Get all available color temperatures."""
        temps = []
        if self.color_temp_k:
            temps.append(self.color_temp_k)
        temps.extend(self.color_temp_options)
        return list(set(temps))
