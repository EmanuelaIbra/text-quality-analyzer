"""
text_redundancy_checker.py
==========================
Detect redundancy, pleonasm, similar words, and synonyms in text.

Requirements:
    pip install spacy nltk

    python -m spacy download en_co

Usage:
    python text_redundancy_checker.py
    -- or edit the TEXT variable at the bottom of this file --
"""

import nltk
import spacy
import pprint
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize

# ---------------------------------------------------------------------------
# Known pleonasms (extend this list as needed)
# ---------------------------------------------------------------------------
PLEONASMS = {
    "free gift", "past history", "future plans", "end result",
    "true fact", "close proximity", "unexpected surprise", "added bonus",
    "advance warning", "final conclusion", "revert back", "repeat again",
    "continue on", "sum total", "new innovation", "safe haven",
    "foreign imports", "terrible tragedy", "completely unanimous",
    "past experience", "first began", "frozen tundra", "hot water heater",
    "join together", "return back", "completely finished", "over and done with",
    "brief moment", "each and every", "basic fundamentals",
}


# ---------------------------------------------------------------------------
# 1. Synonyms via WordNet
# ---------------------------------------------------------------------------
def get_synonyms(word: str) -> set:
    """Return all WordNet synonyms for a word (excluding the word itself)."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    synonyms.discard(word.lower())
    return synonyms


def find_synonyms_in_text(text: str) -> dict:
    """
    For each content word in the text, return its WordNet synonyms.
    Returns a dict: {word: {synonyms}}.
    """
    tokens = word_tokenize(text.lower())
    result = {}
    for token in tokens:
        if token.isalpha() and len(token) > 3:
            syns = get_synonyms(token)
            if syns:
                result[token] = syns
    return result


# ---------------------------------------------------------------------------
# 2. Similar / near-synonym word pairs via spaCy vectors
# ---------------------------------------------------------------------------
def find_similar_words(doc, threshold: float = 0.75) -> list:
    """
    Compare every pair of content words using spaCy word vectors.
    Returns list of (word1, word2, similarity_score) tuples above threshold.
    """
    skip_pos = {"DET", "ADP", "PUNCT", "PRON", "PART", "AUX", "CCONJ", "SCONJ", "SPACE", "NUM"}
    content_tokens = [t for t in doc if t.pos_ not in skip_pos and t.has_vector]

    pairs = []
    for i, t1 in enumerate(content_tokens):
        for t2 in content_tokens[i + 1:]:
            if t1.lemma_.lower() != t2.lemma_.lower():
                sim = t1.similarity(t2)
                if sim > threshold:
                    pairs.append((t1.text, t2.text, round(sim, 2)))

    # Sort by similarity descending, remove duplicates
    seen = set()
    unique = []
    for a, b, s in sorted(pairs, key=lambda x: -x[2]):
        key = frozenset([a.lower(), b.lower()])
        if key not in seen:
            seen.add(key)
            unique.append((a, b, s))
    return unique


# ---------------------------------------------------------------------------
# 3. Redundant sentence pairs via spaCy sentence similarity
# ---------------------------------------------------------------------------
def find_redundant_sentences(sentences: list, docs: list, threshold: float = 0.92) -> list:
    """
    Compare every pair of sentences.
    Returns list of (sentence_a, sentence_b, similarity_score) above threshold.
    """
    redundant = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            sim = docs[i].similarity(docs[j])
            if sim > threshold:
                redundant.append((sentences[i], sentences[j], round(sim, 2)))
    return redundant


# ---------------------------------------------------------------------------
# 4. Pleonasm detection (rule-based)
# ---------------------------------------------------------------------------
def find_pleonasms(text: str) -> list:
    """Return all known pleonastic phrases found in the text."""
    text_lower = text.lower()
    return [p for p in PLEONASMS if p in text_lower]


# ---------------------------------------------------------------------------
# 5. Exact word repetition within the same sentence
# ---------------------------------------------------------------------------
def find_repeated_words(text: str) -> list:
    """
    Flag content words that appear more than once in the same sentence.
    Returns list of dicts: [{"sentence": ..., "words": [...]}]
    """
    results = []
    for sent in sent_tokenize(text):
        tokens = [w for w in word_tokenize(sent.lower()) if w.isalpha() and len(w) > 3]
        counts = {}
        for w in tokens:
            counts[w] = counts.get(w, 0) + 1
        duplicates = [w for w, c in counts.items() if c > 1]
        if duplicates:
            results.append({"sentence": sent, "words": duplicates})
    return results


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def analyze_text(
    text: str,
    word_sim_threshold: float = 0.75,
    sent_sim_threshold: float = 0.92,
    nlp=None,
) -> dict:
    """
    Run all checks on the given text.

    Parameters
    ----------
    text               : The text to analyze.
    word_sim_threshold : Cosine similarity cutoff for word pairs (0–1). Default 0.75.
    sent_sim_threshold : Cosine similarity cutoff for sentence pairs (0–1). Default 0.92.
    nlp                : A loaded spaCy model. Pass one in to avoid reloading.

    Returns
    -------
    dict with keys:
        "pleonasms"           – list of found pleonastic phrases
        "repeated_words"      – list of {sentence, words} dicts
        "similar_words"       – list of (word1, word2, score) tuples
        "redundant_sentences" – list of (sent1, sent2, score) tuples
        "synonyms"            – dict of {word: {synonyms}} from WordNet
    """
    if nlp is None:
        print("Loading spaCy model...")
        nlp = spacy.load("en_core_web_md")

    doc = spacy.tokens.Doc  # just for type hint clarity
    sentences = sent_tokenize(text)
    doc = nlp(text)
    sent_docs = [nlp(s) for s in sentences]

    return {
        "pleonasms":           find_pleonasms(text),
        "repeated_words":      find_repeated_words(text),
        "similar_words":       find_similar_words(doc, threshold=word_sim_threshold),
        "redundant_sentences": find_redundant_sentences(sentences, sent_docs, threshold=sent_sim_threshold),
        "synonyms":            find_synonyms_in_text(text),
    }


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------
def print_report(report: dict) -> None:
    sep = "-" * 60

    print("\n" + sep)
    print("PLEONASMS (unnecessary repetition of meaning)")
    print(sep)
    if report["pleonasms"]:
        for p in report["pleonasms"]:
            print(f"  ⚠  '{p}'")
    else:
        print("  None found.")

    print("\n" + sep)
    print("REPEATED WORDS (same word used twice in one sentence)")
    print(sep)
    if report["repeated_words"]:
        for item in report["repeated_words"]:
            print(f"  Words: {item['words']}")
            print(f"  In:    \"{item['sentence']}\"\n")
    else:
        print("  None found.")

    print("\n" + sep)
    print("SIMILAR / NEAR-SYNONYM WORD PAIRS (spaCy vectors)")
    print(sep)
    if report["similar_words"]:
        for a, b, score in report["similar_words"]:
            bar = "█" * int(score * 20)
            print(f"  '{a}' ↔ '{b}'  {score:.2f}  {bar}")
    else:
        print("  None found.")

    print("\n" + sep)
    print("REDUNDANT SENTENCE PAIRS (same idea, different wording)")
    print(sep)
    if report["redundant_sentences"]:
        for a, b, score in report["redundant_sentences"]:
            print(f"  Similarity: {score:.2f}")
            print(f"  A: {a}")
            print(f"  B: {b}\n")
    else:
        print("  None found.")

    print("\n" + sep)
    print("SYNONYMS PER WORD (WordNet)")
    print(sep)
    for word, syns in list(report["synonyms"].items())[:15]:  # show first 15
        preview = sorted(syns)[:5]
        more = f"  (+{len(syns)-5} more)" if len(syns) > 5 else ""
        print(f"  {word}: {', '.join(preview)}{more}")
    if len(report["synonyms"]) > 15:
        print(f"  ... ({len(report['synonyms']) - 15} more words omitted)")

    print("\n" + sep + "\n")


# ---------------------------------------------------------------------------
# Entry point — edit TEXT here to analyze your own content
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    TEXT = """
    The physician carefully examined the sick patient in the hospital ward.
    The doctor checked the ill person thoroughly at the medical facility.
    We must revert back to our past history to understand the end result.
    The committee made future plans for an unexpected surprise event with an added bonus.
    She will she repeat again her speech at the conference next week.
    The new innovation in renewable energy is a true fact that offers a free gift to society.
    """

    print("Loading spaCy model (en_core_web_md)...")
    nlp_model = spacy.load("en_core_web_md")

    print("Analyzing text...\n")
    report = analyze_text(
        TEXT,
        word_sim_threshold=0.75,   # lower = more pairs found, higher = stricter
        sent_sim_threshold=0.92,   # lower = more sentence pairs flagged
        nlp=nlp_model,
    )

    print_report(report)
