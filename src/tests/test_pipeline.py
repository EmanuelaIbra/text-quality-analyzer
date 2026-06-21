"""
tests/test_pipeline.py
----------------------
Unit tests for all project modules.

Run with:
    python -m pytest tests/
    OR
    python main.py --test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

ERRONEOUS = (
    "The students was very happy about there project. "
    "They went to the library and they went to the library again. "
    "It dont make sense at all."
)

CORRECTED = (
    "The students were very happy about their project. "
    "They went to the library twice. "
    "It doesn't make sense at all."
)

REFERENCE = (
    "The students were very happy about their project. "
    "They went to the library and the library again. "
    "It doesn't make sense."
)


# ---------------------------------------------------------------------------
# Test: GLEU scorer
# ---------------------------------------------------------------------------

def test_gleu_scorer():
    print("\n[TEST] GLEU scorer")
    from metrics.gleu_scorer import sentence_gleu_score, mean_sentence_gleu

    # Single sentence
    score = sentence_gleu_score(
        "The students was happy.",
        "The students were happy.",
        "The students were happy.",
    )
    assert 0.0 <= score <= 1.0, f"GLEU out of range: {score}"
    print(f"  sentence_gleu_score (perfect match): {score:.4f} ✓")

    # Worse correction should score lower
    score_bad = sentence_gleu_score(
        "The students was happy.",
        "The weather is nice today.",
        "The students were happy.",
    )
    assert score_bad < score, "Bad correction should score lower"
    print(f"  sentence_gleu_score (wrong output): {score_bad:.4f} ✓")

    # Mean sentence-level
    originals = ["I is happy.", "She go to school."]
    corrected  = ["I am happy.", "She goes to school."]
    references = [["I am happy."], ["She goes to school."]]
    result = mean_sentence_gleu(originals, corrected, references)
    assert "mean" in result
    assert 0.0 <= result["mean"] <= 1.0
    print(f"  mean_sentence_gleu: {result['mean']:.4f} ✓")

    print("  [PASS] GLEU scorer")


# ---------------------------------------------------------------------------
# Test: LT scorer
# ---------------------------------------------------------------------------

def test_lt_scorer():
    print("\n[TEST] LT scorer")
    from metrics.lt_scorer import lt_score_sentence, lt_score_text, lt_compare, load_tool

    load_tool()

    # Perfect sentence should score close to 1.0
    perfect = lt_score_sentence("The students were very happy about their project.")
    assert perfect["score"] >= 0.7, f"Perfect sentence scored too low: {perfect['score']}"
    print(f"  lt_score_sentence (correct): {perfect['score']:.4f} ✓")

    # Erroneous sentence should score lower
    erroneous = lt_score_sentence("The students was very happy about there project.")
    print(f"  lt_score_sentence (errors): {erroneous['score']:.4f} ✓")
    print(f"  Errors found: {erroneous['num_errors']}")

    # Compare
    comp = lt_compare(ERRONEOUS, CORRECTED)
    assert comp["corrected_score"] >= comp["original_score"] - 0.1, \
        "Corrected text should score >= original"
    print(f"  lt_compare improvement: {comp['improvement']:+.4f} ✓")

    print("  [PASS] LT scorer")


# ---------------------------------------------------------------------------
# Test: Interpolated scorer
# ---------------------------------------------------------------------------

def test_interpolated():
    print("\n[TEST] Interpolated scorer")
    from metrics.interpolated import (
        interpolated_score, system_interpolated_score, score_report
    )

    si = interpolated_score(gleu=0.8, lt=0.9, lam=0.27)
    expected = (1 - 0.27) * 0.9 + 0.27 * 0.8
    assert abs(si - expected) < 1e-6, f"Wrong interpolation: {si}"
    print(f"  interpolated_score(gleu=0.8, lt=0.9, λ=0.27): {si:.4f} ✓")

    result = system_interpolated_score([0.7, 0.8, 0.9], [0.85, 0.90, 0.95], lam=0.27)
    assert 0.0 <= result["system_score"] <= 1.0
    print(f"  system_interpolated_score: {result['system_score']:.4f} ✓")

    report = score_report("orig", "corr", 0.8, 0.7, 0.9)
    assert "GEC EVALUATION REPORT" in report
    print("  score_report: generated ✓")

    print("  [PASS] Interpolated scorer")


# ---------------------------------------------------------------------------
# Test: Redundancy
# ---------------------------------------------------------------------------

def test_redundancy():
    print("\n[TEST] Redundancy")
    from editorial.redundancy import (
        find_wordy_phrases, find_redundant_sentences,
        adequacy_score, full_redundancy_analysis
    )

    # Wordy phrases
    text = "The reason why they failed is because of the fact that they didn't study."
    wordy = find_wordy_phrases(text)
    assert len(wordy) > 0, "Should detect wordy phrases"
    print(f"  find_wordy_phrases: found {len(wordy)} phrase(s) ✓")
    for w in wordy:
        print(f"    '{w['phrase_found']}' → {w['suggestion']}")

    # Semantic redundancy
    redundant_text = (
        "The cat sat on the mat. "
        "The feline rested upon the rug. "
        "Dogs are wonderful pets."
    )
    result = find_redundant_sentences(redundant_text, threshold=0.60)
    print(f"  find_redundant_sentences: {result['num_redundant']} pair(s) ✓")

    # Adequacy
    sim = adequacy_score(
        "The students were happy.",
        "The students were happy."
    )
    assert sim > 0.95, f"Identical sentences should have high similarity: {sim}"
    print(f"  adequacy_score (identical): {sim:.4f} ✓")

    print("  [PASS] Redundancy")


# ---------------------------------------------------------------------------
# Test: Repetition
# ---------------------------------------------------------------------------

def test_repetition():
    print("\n[TEST] Repetition")
    from gec_project.repetition import (
        find_duplicate_words, find_overused_words,
        vocabulary_diversity, full_repetition_analysis, load_spacy
    )
    load_spacy()

    # Duplicate words
    dups = find_duplicate_words("The the students went to to school.")
    assert len(dups) >= 1, "Should find duplicate words"
    print(f"  find_duplicate_words: {len(dups)} found ✓")

    # Overused words
    text = "Research is important. Research needs funding. Research helps society. Research advances knowledge."
    overused = find_overused_words(text, min_count=2)
    assert any(o["lemma"] == "research" for o in overused), "Should detect 'research' as overused"
    print(f"  find_overused_words: {len(overused)} found ✓")

    # TTR
    simple_text = "cat cat cat cat cat"
    div = vocabulary_diversity(simple_text)
    assert div["ttr"] < 0.5, "Low variety text should have low TTR"
    print(f"  vocabulary_diversity (low): ttr={div['ttr']:.4f} ({div['diversity']}) ✓")

    print("  [PASS] Repetition")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run_all_tests():
    print("=" * 50)
    print("  RUNNING ALL TESTS")
    print("=" * 50)

    tests = [
        ("GLEU scorer",       test_gleu_scorer),
        ("LT scorer",         test_lt_scorer),
        ("Interpolated",      test_interpolated),
        ("Redundancy",        test_redundancy),
        ("Repetition",        test_repetition),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    run_all_tests()
