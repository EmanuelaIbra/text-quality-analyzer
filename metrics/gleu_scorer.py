"""
metrics/gleu_scorer.py
----------------------
GLEU (Google-BLEU) scorer — a Reference-Based Metric (RBM).

From the paper:
    "GLEU scores output by penalizing n-grams found in the input and
     output but not the reference." (Napoles et al., 2016)

GLEU is the current state-of-the-art RBM with ρ = 0.852 correlation
with human judgments (Table 1 of the paper).

Usage:
    score = sentence_gleu_score(original, corrected, reference)
    scores = corpus_gleu_scores(originals, corrected_list, references_list)
"""

from nltk.translate.gleu_score import sentence_gleu, corpus_gleu
import nltk

# Make sure NLTK punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return nltk.word_tokenize(text.lower())


# ---------------------------------------------------------------------------
# Sentence-level GLEU  (paper recommends sentence-level, §5)
# ---------------------------------------------------------------------------

def sentence_gleu_score(
    original: str,
    corrected: str,
    reference: str | list[str],
    min_len: int = 1,
    max_len: int = 4,
) -> float:
    """
    Compute GLEU for a single corrected sentence vs one or more references.

    Parameters
    ----------
    original   : the original (possibly erroneous) sentence
    corrected  : the system's corrected output
    reference  : a single reference string OR a list of reference strings.
                 More references → better evaluation (see Figure 1 of paper).
    min_len    : minimum n-gram length (default 1)
    max_len    : maximum n-gram length (default 4)

    Returns
    -------
    float in [0, 1]  — higher is better
    """
    hyp_tokens = _tokenize(corrected)

    # Normalise reference to list-of-token-lists
    if isinstance(reference, str):
        refs = [_tokenize(reference)]
    else:
        refs = [_tokenize(r) for r in reference]

    return sentence_gleu(refs, hyp_tokens, min_len=min_len, max_len=max_len)


# ---------------------------------------------------------------------------
# Corpus-level GLEU  (kept for completeness, paper shows sentence-level wins)
# ---------------------------------------------------------------------------

def corpus_gleu_score(
    originals: list[str],
    corrected_list: list[str],
    references_list: list[list[str]],
) -> float:
    """
    Compute corpus-level GLEU over a full dataset.

    Parameters
    ----------
    originals       : list of original sentences
    corrected_list  : list of corrected sentences (same length)
    references_list : list of reference lists — one list per sentence.
                      e.g. [["ref1a", "ref1b"], ["ref2a"], ...]

    Returns
    -------
    float in [0, 1]
    """
    list_of_refs = [
        [_tokenize(r) for r in refs]
        for refs in references_list
    ]
    hyps = [_tokenize(c) for c in corrected_list]
    return corpus_gleu(list_of_refs, hyps)


# ---------------------------------------------------------------------------
# Mean sentence-level GLEU  (paper's recommended approach §5)
# ---------------------------------------------------------------------------

def mean_sentence_gleu(
    originals: list[str],
    corrected_list: list[str],
    references_list: list[list[str]],
) -> dict:
    """
    Compute the MEAN of sentence-level GLEU scores.
    This is the paper's recommended way to evaluate a system (§5).

    Table 4 of the paper shows corpus-level GLEU = 0.725 but
    sentence-level mean GLEU = 0.852 Spearman ρ with human judgments.

    Returns
    -------
    dict with 'mean', 'scores' (per sentence), 'min', 'max'
    """
    scores = []
    for orig, corr, refs in zip(originals, corrected_list, references_list):
        s = sentence_gleu_score(orig, corr, refs)
        scores.append(s)

    return {
        "mean":   sum(scores) / len(scores) if scores else 0.0,
        "scores": scores,
        "min":    min(scores) if scores else 0.0,
        "max":    max(scores) if scores else 0.0,
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    orig = "The students was very happy about there project."
    corr = "The students were very happy about their project."
    ref  = "The students were very happy about their project."

    score = sentence_gleu_score(orig, corr, ref)
    print(f"GLEU score: {score:.4f}")

    # Multiple references (better evaluation)
    refs_multi = [
        "The students were very happy about their project.",
        "The students were quite happy about their project.",
    ]
    score_multi = sentence_gleu_score(orig, corr, refs_multi)
    print(f"GLEU score (2 refs): {score_multi:.4f}")
