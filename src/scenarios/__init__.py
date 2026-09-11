"""Phase 6: What-if scenario simulation engine.

Re-exports the public API from scenarios.simulator.
"""

from scenarios.simulator import (
    ScenarioResult,
    WhatIfSimulator,
    load_scope_rows,
)

__all__ = [
    "ScenarioResult",
    "WhatIfSimulator",
    "load_scope_rows",
]
