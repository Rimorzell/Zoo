#!/usr/bin/env python3
"""
Command-line interface for the lighting product matching system.

Usage:
    python -m src.cli --requirements input.json --catalog data/product_catalog.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from .matching.matching_engine import (
    MatchingEngine,
    MatchingConfig,
    load_requirements_from_file
)


def main():
    parser = argparse.ArgumentParser(
        description="Lighting Product Matching System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic matching
    python -m src.cli -r input.json -c data/product_catalog.json

    # Output to file
    python -m src.cli -r input.json -c data/product_catalog.json -o results.json

    # Show verbose output
    python -m src.cli -r input.json -c data/product_catalog.json -v

    # Skip validation (faster but less safe)
    python -m src.cli -r input.json -c data/product_catalog.json --skip-validation
        """
    )

    parser.add_argument(
        "-r", "--requirements",
        required=True,
        help="Path to requirements JSON file"
    )
    parser.add_argument(
        "-c", "--catalog",
        required=True,
        help="Path to product catalog JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with detailed matching info"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip source validation (faster but may produce poor matches)"
    )
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="Only output items needing human review"
    )
    parser.add_argument(
        "--format",
        choices=["json", "summary", "csv"],
        default="json",
        help="Output format (default: json)"
    )

    args = parser.parse_args()

    # Validate input files
    req_path = Path(args.requirements)
    catalog_path = Path(args.catalog)

    if not req_path.exists():
        print(f"Error: Requirements file not found: {req_path}", file=sys.stderr)
        sys.exit(1)

    if not catalog_path.exists():
        print(f"Error: Catalog file not found: {catalog_path}", file=sys.stderr)
        sys.exit(1)

    # Configure engine
    config = MatchingConfig(
        skip_validation=args.skip_validation,
    )

    # Initialize engine
    engine = MatchingEngine(config)

    # Load catalog
    if args.verbose:
        print(f"Loading catalog from {catalog_path}...", file=sys.stderr)

    num_products = engine.load_catalog(str(catalog_path))

    if args.verbose:
        print(f"Loaded {num_products} products", file=sys.stderr)

    # Load requirements
    if args.verbose:
        print(f"Loading requirements from {req_path}...", file=sys.stderr)

    requirements = load_requirements_from_file(str(req_path))

    if args.verbose:
        print(f"Processing {len(requirements)} requirements...", file=sys.stderr)

    # Run matching
    summary = engine.match_all(requirements)

    # Generate output
    if args.format == "json":
        output = generate_json_output(summary, engine, args.review_only)
    elif args.format == "summary":
        output = generate_summary_output(summary, engine)
    elif args.format == "csv":
        output = generate_csv_output(summary)

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        if args.verbose:
            print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Print summary to stderr if verbose
    if args.verbose:
        print("\n--- Summary ---", file=sys.stderr)
        print(f"Total: {summary.total_requirements}", file=sys.stderr)
        print(f"Matched: {summary.matched}", file=sys.stderr)
        print(f"Needs Clarification: {summary.needs_clarification}", file=sys.stderr)
        print(f"No Match: {summary.no_match}", file=sys.stderr)
        print(f"Review Required: {summary.review_required}", file=sys.stderr)


def generate_json_output(summary, engine, review_only=False):
    """Generate JSON output."""
    output = summary.to_dict()

    # Add review queue
    output["review_queue"] = engine.get_review_queue(summary)

    # Add metadata
    output["metadata"] = {
        "generated_at": datetime.utcnow().isoformat(),
        "engine_version": "1.0.0",
    }

    if review_only:
        # Filter to only items needing review
        output["matches"] = [
            m for m in output["matches"]
            if m.get("needs_human_review", False)
        ]

    return json.dumps(output, indent=2)


def generate_summary_output(summary, engine):
    """Generate human-readable summary output."""
    lines = []
    lines.append("=" * 60)
    lines.append("LIGHTING PRODUCT MATCHING RESULTS")
    lines.append("=" * 60)
    lines.append("")

    # Summary stats
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total Requirements:    {summary.total_requirements}")
    lines.append(f"Successfully Matched:  {summary.matched}")
    lines.append(f"Needs Clarification:   {summary.needs_clarification}")
    lines.append(f"No Match Found:        {summary.no_match}")
    lines.append(f"Review Required:       {summary.review_required}")
    lines.append("")

    # By confidence level
    lines.append("BY CONFIDENCE LEVEL")
    lines.append("-" * 40)
    for level, count in sorted(summary.by_confidence_level.items()):
        lines.append(f"  {level}: {count}")
    lines.append("")

    # Matches
    lines.append("MATCH DETAILS")
    lines.append("-" * 40)

    for match in summary.matches:
        status_icon = {
            "AUTO_APPROVE": "[OK]",
            "REVIEW_RECOMMENDED": "[?]",
            "REVIEW_REQUIRED": "[!]",
            "NEEDS_CLARIFICATION": "[??]",
            "NO_MATCH": "[X]",
        }.get(match.confidence_level.value, "[ ]")

        lines.append(f"\n{status_icon} {match.line_item} (qty: {match.quantity})")
        lines.append(f"    Confidence: {match.confidence:.0%} ({match.confidence_level.value})")

        if match.matched_sku:
            lines.append(f"    Matched: {match.matched_sku}")
            lines.append(f"             {match.matched_product_name}")
        else:
            lines.append(f"    Matched: None")

        lines.append(f"    Reasoning: {match.reasoning[:80]}...")

        if match.warnings:
            lines.append(f"    Warnings:")
            for w in match.warnings[:3]:
                lines.append(f"      - {w}")

        if match.alternatives:
            lines.append(f"    Alternatives:")
            for alt in match.alternatives[:2]:
                lines.append(f"      - {alt.sku} ({alt.confidence:.0%})")

    # Review queue
    review_queue = engine.get_review_queue(summary)
    if review_queue:
        lines.append("")
        lines.append("=" * 60)
        lines.append("ITEMS REQUIRING REVIEW")
        lines.append("=" * 60)

        for item in review_queue:
            priority_icon = {1: "!!!", 2: "!!", 3: "!", 4: "?", 5: ""}.get(item["priority"], "")
            lines.append(f"\n{priority_icon} Priority {item['priority']}: {item['line_item']}")
            lines.append(f"   Level: {item['level']}")
            lines.append(f"   Action: {item['action_needed']}")
            if item.get("matched_sku"):
                lines.append(f"   Current Match: {item['matched_sku']} ({item['confidence']:.0%})")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def generate_csv_output(summary):
    """Generate CSV output."""
    lines = []

    # Header
    lines.append(",".join([
        "line_item",
        "quantity",
        "unit",
        "matched_sku",
        "matched_product_name",
        "confidence",
        "confidence_level",
        "needs_review",
        "review_reason",
    ]))

    # Data rows
    for match in summary.matches:
        row = [
            match.line_item,
            str(match.quantity),
            match.unit,
            match.matched_sku or "",
            match.matched_product_name or "",
            f"{match.confidence:.2f}",
            match.confidence_level.value,
            "Yes" if match.needs_human_review else "No",
            (match.review_reason or "").replace(",", ";"),
        ]
        lines.append(",".join(row))

    return "\n".join(lines)


if __name__ == "__main__":
    main()
