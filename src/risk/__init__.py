"""Phase 5: Risk Scoring Engine.

Combines Phase 4 ML delay predictions with Phase 3 CPM results to produce
activity-level risk scores.

Risk components (all clearly separated):
    1. probability_of_event_delay  — LogisticRegression predict_proba output (TRUE probability)
    2. predicted_delay_days        — LinearRegression prediction (predicted magnitude)
    3. criticality_weight          — Heuristic based on total_float bins
    4. float_impact                — Heuristic: predicted_delay / available_float
    5. risk_score                  — probability * criticality_weight * float_impact

Heuristic components (criticality_weight, float_impact) are documented as
heuristics, NOT learned/statistical facts.

The production risk engine does NOT depend on validation-set predictions specifically.
"""

from risk.scoring import (
    RiskScoreCalculator,
    calculate_risk_scores,
    criticality_weight,
    float_impact,
)

__all__ = [
    "RiskScoreCalculator",
    "calculate_risk_scores",
    "criticality_weight",
    "float_impact",
]
