#!/usr/bin/env python3
"""
Phase 5: Risk Scoring Execution Script.

Calculates activity-level risk scores by combining:
1. Phase 4 ML delay predictions (LogisticRegression probability, LinearRegression magnitude)
2. Phase 3 CPM total float / criticality
3. Heuristic criticality weight and float impact

Output:
    data/processed/risk_scores.csv

IMPORTANT:
- Uses true LogisticRegression probability (predict_proba), NOT normalized scores
- Criticality weight and float impact are HEURISTICS, documented as such
- Risk score = probability * criticality_weight * float_impact
- Test set remains locked; no test data used for risk score design
- probability_of_event_delay is the TRUE probability of event-induced delay
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from risk.scoring import RiskScoreCalculator, calculate_risk_scores


def main() -> None:
    """Run Phase 5 risk scoring."""
    print("=" * 70)
    print("PHASE 5: RISK SCORING CALCULATION")
    print("=" * 70)

    # Initialize calculator
    calculator = RiskScoreCalculator()

    # Load CPM results
    print("\n1. Loading CPM results...")
    calculator.load_cpm_results()
    print(f"   Loaded {len(calculator.cpm_results)} activities")

    # Load activities metadata (optional)
    print("\n2. Loading activities metadata...")
    calculator.load_activities()
    if calculator.activities is not None and len(calculator.activities) > 0:
        print(f"   Loaded {len(calculator.activities)} activities")
    else:
        print("   No activities metadata available")

    # Calculate risk scores using the full Phase 4 pipeline
    print("\n3. Calculating risk scores...")

    from ml.modeling import load_step2_as_dataframes, build_preprocessed_split

    processed_dir = calculator.processed_dir
    raw_split = load_step2_as_dataframes(processed_dir)
    preprocessed = build_preprocessed_split(raw_split, fit_preprocessor_now=True)

    results = calculator.calculate(preprocessed, fit_models_now=True)

    print(f"   Risk scores calculated for {len(results)} activities")

    # Print summary
    print("\n4. Risk Score Summary:")
    print("-" * 40)

    scores = results["risk_score"]
    print(f"   Total activities: {len(results)}")
    print(f"   Mean risk score: {scores.mean():.4f}")
    print(f"   Std risk score: {scores.std():.4f}")
    print(f"   Min risk score: {scores.min():.4f}")
    print(f"   Max risk score: {scores.max():.4f}")

    q75 = float(scores.quantile(0.75))
    q95 = float(scores.quantile(0.95))
    print(f"   High-risk threshold (75th pctl): {q75:.4f}")
    print(f"   Critical-risk threshold (95th pctl): {q95:.4f}")
    print(f"   High-risk activities: {(scores > q75).sum()}")
    print(f"   Critical-risk activities: {(scores > q95).sum()}")

    print("\n5. Top 10 Highest-Risk Activities:")
    print("-" * 40)

    display_cols = [
        "activity_id",
        "project_id",
        "risk_score",
        "risk_rank",
        "probability_of_event_delay",
        "predicted_delay_days",
        "total_float",
        "criticality_weight",
        "float_impact",
    ]

    available_cols = [c for c in display_cols if c in results.columns]
    print(results[available_cols].head(10).to_string(index=False))

    # Save results
    output_path = calculator.processed_dir / "risk_scores.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\n6. Results saved to: {output_path}")

    print("\n" + "=" * 70)
    print("PHASE 5 RISK SCORING COMPLETE")
    print("=" * 70)

    print("\nNOTE: probability_of_event_delay is the TRUE probability from "
          "LogisticRegression.predict_proba(), NOT a normalized score.")
    print("NOTE: criticality_weight and float_impact are HEURISTICS, not learned "
          "or statistical quantities.")
    print("NOTE: risk_score = probability * criticality_weight * float_impact")


if __name__ == "__main__":
    main()
