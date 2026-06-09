"""
metrics/interpolated.py
-----------------------
Interpolated metric: combines a Reference-Based Metric (RBM) with a
Grammaticality-Based Metric (GBM).

From the paper (§3.1):
    SI = (1 − λ) * SG  +  λ * SR

Where:
    SI = interpolated score
    SG = GBM score  (Language Tool, reference-less)
    SR = RBM score  (GLEU, reference-based)
    λ  = mixing weight in [0, 1]

Oracle λ values from Table 2:
    GLEU + LT:  λ = 0.27  →  ρ = 0.874  (best open-source combo)
    GLEU + ER:  λ = 0.04  →  ρ = 0.885  (best overall, but ER is proprietary)

The paper's key finding: interpolation ALWAYS beats either metric alone.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Single-sentence interpolated score
# ---------------------------------------------------------------------------

def interpolated_score(
    gleu: float,
    lt: float,
    lam: float = 0.27,
) -> float:
    """
    Compute the interpolated score for a single sentence.

    Parameters
    ----------
    gleu : GLEU score (SR — reference-based)   in [0, 1]
    lt   : LT score  (SG — grammaticality)      in [0, 1]
    lam  : λ mixing weight (default 0.27, oracle from paper Table 2)

    Returns
    -------
    float in [0, 1]
    """
    return (1 - lam) * lt + lam * gleu


# ---------------------------------------------------------------------------
# System-level score (mean of sentence-level interpolated scores)
# ---------------------------------------------------------------------------

def system_interpolated_score(
    gleu_scores: list[float],
    lt_scores: list[float],
    lam: float = 0.27,
) -> dict:
    """
    Compute the system-level interpolated score as the MEAN of
    sentence-level interpolated scores. (Paper §5 recommendation.)

    Parameters
    ----------
    gleu_scores : list of per-sentence GLEU scores
    lt_scores   : list of per-sentence LT scores
    lam         : λ mixing weight

    Returns
    -------
    dict:
        system_score       : float — final system quality score
        sentence_scores    : list[float]
        gleu_mean          : float
        lt_mean            : float
        lambda_used        : float
    """
    assert len(gleu_scores) == len(lt_scores), \
        "gleu_scores and lt_scores must have the same length"

    sentence_scores = [
        interpolated_score(g, l, lam)
        for g, l in zip(gleu_scores, lt_scores)
    ]

    return {
        "system_score":    np.mean(sentence_scores),
        "sentence_scores": sentence_scores,
        "gleu_mean":       np.mean(gleu_scores),
        "lt_mean":         np.mean(lt_scores),
        "lambda_used":     lam,
    }


# ---------------------------------------------------------------------------
# Lambda search — find the best λ for a dataset with human scores
# ---------------------------------------------------------------------------

def find_best_lambda(
    gleu_scores: list[float],
    lt_scores: list[float],
    human_scores: list[float],
    n_steps: int = 101,
) -> dict:
    """
    Search for the oracle λ that maximises Spearman correlation with
    human judgments. (Replicates the paper's Table 2 experiment.)

    Parameters
    ----------
    gleu_scores  : per-system GLEU scores
    lt_scores    : per-system LT scores
    human_scores : per-system human quality scores
    n_steps      : number of λ values to test (default 101 = 0.00 to 1.00)

    Returns
    -------
    dict:
        best_lambda    : float
        best_spearman  : float
        all_lambdas    : list[float]
        all_spearmans  : list[float]
    """
    from scipy.stats import spearmanr

    lambdas = np.linspace(0, 1, n_steps)
    spearmans = []

    for lam in lambdas:
        interp = [
            interpolated_score(g, l, lam)
            for g, l in zip(gleu_scores, lt_scores)
        ]
        rho, _ = spearmanr(interp, human_scores)
        spearmans.append(rho)

    best_idx = int(np.argmax(spearmans))

    return {
        "best_lambda":   float(lambdas[best_idx]),
        "best_spearman": float(spearmans[best_idx]),
        "all_lambdas":   lambdas.tolist(),
        "all_spearmans": spearmans,
    }


# ---------------------------------------------------------------------------
# Score normalisation helpers (needed because I-measure is in [-1,1])
# ---------------------------------------------------------------------------

def normalise_score(score: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """Normalise any score to [0, 1] range."""
    if max_val == min_val:
        return 0.5
    return (score - min_val) / (max_val - min_val)


# ---------------------------------------------------------------------------
# Summary reporter
# ---------------------------------------------------------------------------

def score_report(
    original: str,
    corrected: str,
    gleu: float,
    lt_original: float,
    lt_corrected: float,
    lam: float = 0.27,
) -> str:
    """
    Return a formatted string report of all metric scores.
    """
    si = interpolated_score(gleu, lt_corrected, lam)
    lines = [
        "=" * 50,
        "  GEC EVALUATION REPORT",
        "=" * 50,
        "",
        "  Reference-Based Metric (RBM)",
        f"    GLEU score          : {gleu:.4f}",
        "",
        "  Grammaticality-Based Metric (GBM)",
        f"    LT score (original) : {lt_original:.4f}",
        f"    LT score (corrected): {lt_corrected:.4f}",
        f"    Improvement         : {lt_corrected - lt_original:+.4f}",
        "",
        "  Interpolated Score (paper §3.1)",
        f"    SI = (1-{lam}) * LT + {lam} * GLEU",
        f"    SI                  : {si:.4f}",
        "=" * 50,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate scores for 3 sentences
    gleu_scores = [0.72, 0.85, 0.60]
    lt_scores   = [0.90, 0.95, 0.80]

    result = system_interpolated_score(gleu_scores, lt_scores, lam=0.27)
    print(f"System score (λ=0.27): {result['system_score']:.4f}")
    print(f"GLEU mean:             {result['gleu_mean']:.4f}")
    print(f"LT mean:               {result['lt_mean']:.4f}")
    print(f"Per-sentence scores:   {[round(s, 4) for s in result['sentence_scores']]}")
