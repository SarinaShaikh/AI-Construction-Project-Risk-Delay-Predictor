"""Phase 6: What-if scenario simulation engine.

Re-exports the public API from scenarios.simulator.
"""

# Package-relative import (branch phase7-integration import fix, same class
# of bug as src/risk/__init__.py: top-level 'from scenarios.simulator ...'
# breaks when imported as src.scenarios).
from .simulator import (
    ScenarioResult,
    WhatIfSimulator,
    load_scope_rows,
)

__all__ = [
    "ScenarioResult",
    "WhatIfSimulator",
    "load_scope_rows",
]
