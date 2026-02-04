"""LLM-based intelligent product selection.

When multiple candidates pass hard constraints and have similar scores,
uses LLM reasoning to select the best match.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json

from ..models.requirement import Requirement
from ..models.product import Product
from .preference_scorer import ScoredCandidate


@dataclass
class SelectionResult:
    """Result of LLM-based selection."""
    selected_sku: str
    selected_name: str
    confidence: float
    reasoning: str
    key_match_points: List[str]
    trade_offs: List[str]
    alternatives: List[Dict]
    warnings: List[str]

    def to_dict(self) -> Dict:
        return {
            "selected_sku": self.selected_sku,
            "selected_name": self.selected_name,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_match_points": self.key_match_points,
            "trade_offs": self.trade_offs,
            "alternatives": self.alternatives,
            "warnings": self.warnings,
        }


class LLMSelector:
    """Intelligent product selection using LLM reasoning.

    This class provides structured prompts and selection logic that can be
    used with any LLM backend. The actual LLM call can be injected or the
    rule-based fallback can be used.
    """

    # Threshold for when LLM selection is needed
    SCORE_DIFF_THRESHOLD = 0.05  # If top candidates within 5%, use LLM
    MIN_CANDIDATES_FOR_LLM = 2

    def __init__(self, llm_backend: Optional[Any] = None):
        """
        Initialize selector.

        Args:
            llm_backend: Optional LLM backend for API calls.
                        If None, uses rule-based selection.
        """
        self.llm_backend = llm_backend

    def needs_llm_selection(self, scored_candidates: List[ScoredCandidate]) -> bool:
        """Determine if LLM selection is needed."""
        if len(scored_candidates) < self.MIN_CANDIDATES_FOR_LLM:
            return False

        # Check if top candidates are close in score
        if len(scored_candidates) >= 2:
            top_score = scored_candidates[0].score
            second_score = scored_candidates[1].score
            return (top_score - second_score) <= self.SCORE_DIFF_THRESHOLD

        return False

    def select(
        self,
        requirement: Requirement,
        scored_candidates: List[ScoredCandidate]
    ) -> SelectionResult:
        """
        Select the best product from scored candidates.

        Args:
            requirement: The requirement to match
            scored_candidates: Scored candidates sorted by score descending

        Returns:
            SelectionResult with selection and reasoning
        """
        if not scored_candidates:
            return SelectionResult(
                selected_sku="",
                selected_name="",
                confidence=0.0,
                reasoning="No candidates available for selection",
                key_match_points=[],
                trade_offs=[],
                alternatives=[],
                warnings=["No products matched the requirements"],
            )

        # If only one candidate or clear winner, use rule-based selection
        if not self.needs_llm_selection(scored_candidates):
            return self._rule_based_selection(requirement, scored_candidates)

        # Try LLM selection if backend available
        if self.llm_backend:
            try:
                return self._llm_selection(requirement, scored_candidates)
            except Exception as e:
                # Fall back to rule-based on error
                result = self._rule_based_selection(requirement, scored_candidates)
                result.warnings.append(f"LLM selection failed, used rule-based: {str(e)}")
                return result

        # Default to rule-based
        return self._rule_based_selection(requirement, scored_candidates)

    def _rule_based_selection(
        self,
        requirement: Requirement,
        scored_candidates: List[ScoredCandidate]
    ) -> SelectionResult:
        """Select best product using rule-based logic."""
        best = scored_candidates[0]
        alternatives = []

        # Build alternatives list
        for candidate in scored_candidates[1:4]:  # Top 3 alternatives
            alternatives.append({
                "sku": candidate.product.sku,
                "name": candidate.product.name,
                "confidence": round(candidate.score, 2),
                "note": "; ".join(candidate.notes) if candidate.notes else "Alternative match",
            })

        # Build reasoning
        reasoning_parts = []
        reasoning_parts.append(f"Selected {best.product.name} as best match.")

        if best.breakdown.reference_score > 0.8:
            reasoning_parts.append("Product is compatible with reference specification.")
        if best.breakdown.wattage_score > 0.9:
            reasoning_parts.append("Wattage closely matches target.")
        if best.breakdown.lumens_score > 0.9:
            reasoning_parts.append("Lumens output matches requirement.")
        if best.breakdown.color_temp_score > 0.9:
            reasoning_parts.append("Color temperature matches specification.")

        # Identify key match points
        key_points = []
        if requirement.requirements.wattage and best.product.wattage:
            key_points.append(f"Wattage: {best.product.wattage}W matches target {requirement.requirements.wattage.target}W")
        if requirement.requirements.lumens and best.product.lumens:
            key_points.append(f"Lumens: {best.product.lumens}lm meets requirement")
        if best.product.ip_rating and requirement.requirements.min_ip_rating:
            key_points.append(f"IP{best.product.ip_rating} meets IP{requirement.requirements.min_ip_rating} requirement")

        # Identify trade-offs
        trade_offs = []
        if best.notes:
            trade_offs.extend(best.notes)

        # Calculate confidence
        # Single candidate = score is confidence
        # Multiple close candidates = reduce confidence
        confidence = best.score
        if len(scored_candidates) >= 2:
            second_score = scored_candidates[1].score
            score_diff = best.score - second_score
            if score_diff < 0.03:
                confidence = min(confidence, 0.85)
                trade_offs.append("Multiple very similar options available")
            elif score_diff < 0.10:
                confidence = min(confidence, 0.90)

        # Collect warnings
        warnings = []
        if best.notes:
            for note in best.notes:
                if any(term in note.lower() for term in ["differs", "exceeds", "below", "not supported"]):
                    warnings.append(note)

        return SelectionResult(
            selected_sku=best.product.sku,
            selected_name=best.product.name,
            confidence=round(confidence, 2),
            reasoning=" ".join(reasoning_parts),
            key_match_points=key_points,
            trade_offs=trade_offs,
            alternatives=alternatives,
            warnings=warnings,
        )

    def _llm_selection(
        self,
        requirement: Requirement,
        scored_candidates: List[ScoredCandidate]
    ) -> SelectionResult:
        """Select best product using LLM reasoning."""
        prompt = self._build_selection_prompt(requirement, scored_candidates)

        # Call LLM backend
        response = self.llm_backend.complete(prompt)

        # Parse response
        return self._parse_llm_response(response, scored_candidates)

    def _build_selection_prompt(
        self,
        requirement: Requirement,
        scored_candidates: List[ScoredCandidate]
    ) -> str:
        """Build the prompt for LLM selection."""
        # Build requirement summary
        req_summary = {
            "line_item": requirement.line_item,
            "description": requirement.description,
            "product_type": requirement.requirements.product_type,
            "mounting": requirement.requirements.mounting,
            "wattage": requirement.requirements.wattage.__dict__ if requirement.requirements.wattage else None,
            "lumens": requirement.requirements.lumens.__dict__ if requirement.requirements.lumens else None,
            "color_temp_k": requirement.requirements.color_temp_k.__dict__ if requirement.requirements.color_temp_k else None,
            "min_ip_rating": requirement.requirements.min_ip_rating,
            "dimensions_mm": requirement.requirements.dimensions_mm,
            "emergency_type": requirement.requirements.emergency_type,
            "reference_catalog": requirement.requirements.reference_catalog,
        }

        # Build candidate summaries
        candidates_summary = []
        for sc in scored_candidates[:5]:  # Top 5 candidates
            candidates_summary.append({
                "sku": sc.product.sku,
                "name": sc.product.name,
                "score": round(sc.score, 3),
                "wattage": sc.product.wattage,
                "lumens": sc.product.lumens,
                "color_temp_k": sc.product.color_temp_k,
                "ip_rating": sc.product.ip_rating,
                "dimensions_mm": sc.product.dimensions_mm,
                "compatible_with": sc.product.compatible_with,
                "notes": sc.notes,
            })

        prompt = f"""TASK: Select the best matching product for this lighting requirement.

REQUIREMENT:
{json.dumps(req_summary, indent=2)}

CANDIDATES (all pass hard constraints, sorted by preference score):
{json.dumps(candidates_summary, indent=2)}

SELECTION RULES:
1. All candidates are technically acceptable (passed hard constraints)
2. Select the BEST match based on:
   - Closest match to target wattage and lumens
   - Color temperature alignment
   - Similarity to reference product (if specified)
   - Feature compatibility (dimming, emergency options)
   - Overall suitability for the application

3. Provide confidence score:
   - 0.95-1.0: Near-perfect match
   - 0.85-0.94: Good match with minor differences
   - 0.70-0.84: Acceptable match with notable trade-offs
   - Below 0.70: Uncertain, needs human verification

4. Identify alternatives that are also acceptable
5. Note any warnings or concerns

RESPOND WITH JSON:
{{
  "selected_sku": "best match SKU",
  "confidence": 0.XX,
  "reasoning": "clear explanation",
  "key_match_points": ["list", "of", "why", "this", "matches"],
  "trade_offs": ["list", "of", "differences", "from", "ideal"],
  "alternatives": [{{"sku": "...", "confidence": 0.XX, "note": "..."}}],
  "warnings": ["any", "concerns"]
}}"""

        return prompt

    def _parse_llm_response(
        self,
        response: str,
        scored_candidates: List[ScoredCandidate]
    ) -> SelectionResult:
        """Parse LLM response into SelectionResult."""
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                # Find the selected product
                selected_sku = data.get("selected_sku", "")
                valid_skus = {sc.product.sku for sc in scored_candidates}
                if selected_sku not in valid_skus:
                    raise ValueError("Selected SKU not in candidates")
                selected_product = None
                for sc in scored_candidates:
                    if sc.product.sku == selected_sku:
                        selected_product = sc.product
                        break

                confidence = float(data.get("confidence", 0.7))
                if not 0.0 <= confidence <= 1.0:
                    confidence = min(max(confidence, 0.0), 1.0)

                return SelectionResult(
                    selected_sku=selected_sku,
                    selected_name=selected_product.name if selected_product else "",
                    confidence=confidence,
                    reasoning=data.get("reasoning", ""),
                    key_match_points=data.get("key_match_points", []),
                    trade_offs=data.get("trade_offs", []),
                    alternatives=data.get("alternatives", []),
                    warnings=data.get("warnings", []),
                )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Return fallback result
            pass

        # Fallback to rule-based if parsing fails
        best = scored_candidates[0]
        return SelectionResult(
            selected_sku=best.product.sku,
            selected_name=best.product.name,
            confidence=best.score * 0.9,  # Reduce confidence due to parsing failure
            reasoning="Selected top-scored candidate (LLM response parsing failed)",
            key_match_points=[],
            trade_offs=["LLM response could not be parsed"],
            alternatives=[],
            warnings=["LLM selection parsing failed, used fallback"],
        )

    def get_selection_prompt(
        self,
        requirement: Requirement,
        scored_candidates: List[ScoredCandidate]
    ) -> str:
        """
        Get the selection prompt for external LLM usage.

        Useful when the LLM call is made externally.
        """
        return self._build_selection_prompt(requirement, scored_candidates)
