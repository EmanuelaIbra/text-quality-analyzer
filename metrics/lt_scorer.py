"""
metrics/lt_scorer.py
--------------------
Language Tool (LT) scorer — a Grammaticality-Based Metric (GBM).
This is the paper's open-source, reference-less metric.

From the paper:
    "The error count score is simply calculated: 1 − #errors / #tokens.
     Language Tool is publicly available and open sourced."

Results from Table 1:
    LT alone:          Spearman ρ = 0.808
    GLEU + LT (λ=0.27): Spearman ρ = 0.874  (beats standalone GLEU!)

This module requires Java to be installed (Language Tool runs on JVM).
Install: pip install language-tool-python
"""

import language_tool_python
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ---------------------------------------------------------------------------
# Tool loading (lazy singleton)
# ---------------------------------------------------------------------------

_tool = None


def load_tool(language: str = "en-US"):
    """Load Language Tool. Call once at startup."""
    global _tool
    print("Loading Language Tool (requires Java) ...")
    _tool = language_tool_python.LanguageTool(language)
    print("Language Tool loaded.")


def _ensure_loaded():
    if _tool is None:
        load_tool()


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def lt_score_sentence(sentence: str) -> dict:
    """
    Score a single sentence using Language Tool.

    Paper formula: score = 1 - (num_errors / num_tokens)

    Returns
    -------
    dict:
        score       : float in [0, 1]  — higher = more grammatical
        num_errors  : int
        num_tokens  : int
        errors      : list of error dicts (message, category, suggestions)
    """
    _ensure_loaded()

    tokens = nltk.word_tokenize(sentence)
    num_tokens = len(tokens)
    if num_tokens == 0:
        return {"score": 1.0, "num_errors": 0, "num_tokens": 0, "errors": []}

    matches = _tool.check(sentence)
    num_errors = len(matches)

    score = 1.0 - (num_errors / num_tokens)
    score = max(0.0, score)  # clamp to [0, 1]

    errors = []
    for m in matches:
        errors.append({
            "message":     m.message,
            "category":    m.category,
            "context":     m.context,
            "offset":      m.offset,
            "length": getattr(m, "errorLength", getattr(m, "length", 0)),
            "suggestions": list(m.replacements[:3]),  # top 3 suggestions
        })

    return {
        "score":      score,
        "num_errors": num_errors,
        "num_tokens": num_tokens,
        "errors":     errors,
    }


def lt_score_text(text: str) -> dict:
    """
    Score a full text by computing the MEAN of sentence-level LT scores.
    (Paper §5: sentence-level mean is more reliable than corpus-level.)

    Returns
    -------
    dict:
        mean_score      : float — system-level score
        sentence_scores : list[float]
        sentence_details: list[dict] — full detail per sentence
        total_errors    : int
    """
    _ensure_loaded()
    sentences = nltk.sent_tokenize(text)
    details = []
    scores = []

    for sent in sentences:
        result = lt_score_sentence(sent)
        details.append(result)
        scores.append(result["score"])

    mean_score = sum(scores) / len(scores) if scores else 1.0
    total_errors = sum(d["num_errors"] for d in details)

    return {
        "mean_score":       mean_score,
        "sentence_scores":  scores,
        "sentence_details": details,
        "total_errors":     total_errors,
        "num_sentences":    len(sentences),
    }


def lt_compare(original: str, corrected: str) -> dict:
    """
    Compare LT scores before and after correction.
    Useful for showing the improvement to a user.

    Returns
    -------
    dict with original_score, corrected_score, improvement, errors_removed
    """
    orig_result = lt_score_text(original)
    corr_result = lt_score_text(corrected)

    improvement = corr_result["mean_score"] - orig_result["mean_score"]
    errors_removed = orig_result["total_errors"] - corr_result["total_errors"]

    return {
        "original_score":  orig_result["mean_score"],
        "corrected_score": corr_result["mean_score"],
        "improvement":     improvement,
        "errors_removed":  errors_removed,
        "original_detail": orig_result,
        "corrected_detail": corr_result,
    }


# ---------------------------------------------------------------------------
# Error categorisation helper
# ---------------------------------------------------------------------------

def categorise_errors(lt_result: dict) -> dict:
    """
    Group errors from lt_score_text() by category.
    Returns a dict: {category: count}
    """
    categories = {}
    for sent_detail in lt_result["sentence_details"]:
        for err in sent_detail["errors"]:
            cat = err["category"]
            categories[cat] = categories.get(cat, 0) + 1
    return dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    original  = "The students was very happy about there project. It dont make sense."
    corrected = "The students were very happy about their project. It doesn't make sense."

    print("=== Original ===")
    orig_result = lt_score_text(original)
    print(f"LT score: {orig_result['mean_score']:.4f}  |  errors: {orig_result['total_errors']}")

    print("\n=== Corrected ===")
    corr_result = lt_score_text(corrected)
    print(f"LT score: {corr_result['mean_score']:.4f}  |  errors: {corr_result['total_errors']}")

    print("\n=== Comparison ===")
    comp = lt_compare(original, corrected)
    print(f"Improvement: {comp['improvement']:+.4f}")
    print(f"Errors removed: {comp['errors_removed']}")
