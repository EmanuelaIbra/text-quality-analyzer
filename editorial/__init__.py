"""
Editorial package
-----------------
Layered NLP system:

Layer 1:
    Grammar correction

Layer 2:
    Structural redundancy (regex + syntax-safe fixes)

Layer 3:
    Information-theoretic redundancy (entropy-based compression)

Layer 2B:
    Repetition analysis (report only)
"""

from .redundancy import (
    layer2_redundancy,
    apply_exact_patterns,
    apply_pronoun_fixes,
)

from .redundancy_metrics import (
    redundancy_pipeline,
)

from ..repetition import (
    full_repetition_analysis,
)

__all__ = [
    # Layer 2
    "layer2_redundancy",
    "apply_exact_patterns",
    "apply_pronoun_fixes",

    # Layer 3 (NEW MATH ENGINE)
    "redundancy_pipeline",

    # Layer 2B
    "full_repetition_analysis",
]