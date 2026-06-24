import json
import os
import re
import spacy
import numpy as np
from functools import lru_cache
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# POS tags considered meaningful for repetition/similarity checks.
SEMANTIC_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

# Thresholds for redundancy classification (transformer cosine scores)
REDUNDANCY_THRESHOLDS = {
    "duplicate":       0.95,
    "manual_review":   0.85,
    "merge_candidate": 0.70,
}

# ---------------------------------------------------------------------------
# Sentence-transformer model (loaded once, shared everywhere)
# ---------------------------------------------------------------------------

# Best free multilingual model for Italian paraphrase detection.
# normalize_embeddings=True means cosine similarity == dot product (faster).
def encode_sentences(sentences: list[str]) -> np.ndarray:
    """Batch-encode sentences, release model from memory after use."""
    # Load the multilingual paraphrase model on CPU for compatibility.
    model = SentenceTransformer(
        "paraphrase-multilingual-mpnet-base-v2",
        device='cpu'
    )
    # Encode all sentences in batches and normalize vectors for cosine similarity.
    result = model.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    # Release model from memory
    del model
    torch.cuda.empty_cache()
    return result

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

# Normalizes whitespace and punctuation spacing before linguistic analysis.
def normalize_spacing(text: str) -> str:
    # Collapse repeated whitespace into a single space.
    text = re.sub(r"\s+", " ", text)
    # Add a missing space after sentence-ending punctuation when followed by uppercase text.
    text = re.sub(r"(?<=[.!?])(?=[A-ZÀ-Ü])", " ", text)
    # Remove unnecessary spaces before punctuation marks.
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text.strip()

# Produces a simplified sentence form for exact duplicate comparison.
def normalize_for_dedup(sentence: str) -> str:
    # Lowercase and trim to make comparisons case-insensitive.
    sentence = sentence.lower().strip()
    # Remove punctuation while preserving accented Italian characters.
    sentence = re.sub(r"[^\wà-ÿ\s]", "", sentence)
    return re.sub(r"\s+", " ", sentence).strip()


# Splits normalized text into sentence-like units using punctuation boundaries.
def split_sentences(text: str) -> list[str]:
    text = normalize_spacing(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

# ---------------------------------------------------------------------------
# Pleonasm loading & detection
# ---------------------------------------------------------------------------

# Loads pleonasm definitions from the configured JSON dataset.
def load_pleonasm_entries(json_path: str = "data/italian_pleonasms.json") -> list[dict]:
    if not os.path.exists(json_path):
        print(f"Warning: pleonasm file not found: {json_path}")
        return []

     # Read JSON using UTF-8 because the data contains Italian accented text.
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for category_name, category_data in data.get("categories", {}).items():
        for item in category_data.get("examples", []):
            phrase = item.get("pleonasmo", "").strip()
            if not phrase:
                continue
            entries.append({
                "phrase":          phrase,
                "replacement":     item.get("forma_corretta", "").strip(),
                "category":        category_name,
                "explanation":     item.get("spiegazione", ""),
                "correct_variant": item.get("variante_corretta", ""),
            })

    return entries


# Load pleonasm entries once at module import time.
PLEONASM_ENTRIES = load_pleonasm_entries()
# Cache lemma-based patterns per spaCy model name to avoid rebuilding them repeatedly.
_PATTERN_CACHE: dict = {}


# Converts pleonasm phrases into lemma sequences for inflection-tolerant matching.
def _build_lemma_patterns(entries: list[dict], nlp) -> list[dict]:
    patterns = []
    for entry in entries:
         # Lemmatize the phrase and ignore punctuation/space tokens.
        lemmas = [
            t.lemma_.lower()
            for t in nlp(entry["phrase"])
            if not t.is_punct and not t.is_space and t.text.strip()
        ]
        if lemmas:
            patterns.append({**entry, "phrase": entry["phrase"].lower(), "lemmas": lemmas})
    return patterns


# Retrieves cached lemma patterns for the current spaCy pipeline.
def get_lemma_patterns(nlp) -> list[dict]:
    key = nlp.meta.get("name", "default")
    if key not in _PATTERN_CACHE:
        _PATTERN_CACHE[key] = _build_lemma_patterns(PLEONASM_ENTRIES, nlp)
    return _PATTERN_CACHE[key]

# Optional helper to prebuild pleonasm patterns before the first analysis call.
def warmup_pleonasm_cache(nlp) -> None:
    get_lemma_patterns(nlp)

# Finds pleonastic expressions by comparing token lemmas with stored patterns.
def find_pleonasms(text: str, nlp) -> list[dict]:
    text         = normalize_spacing(text)
    doc          = nlp(text)
    # Keep only meaningful tokens so pattern matching is not affected by punctuation
    tokens       = [t for t in doc if not t.is_punct and not t.is_space and t.text.strip()]
    token_lemmas = [t.lemma_.lower() for t in tokens]
  # seen prevents reporting the same match more than once.
    findings, seen = [], set()

    for pattern in get_lemma_patterns(nlp):
        size   = len(pattern["lemmas"])
        target = pattern["lemmas"]

        for i in range(len(token_lemmas) - size + 1):
             # Skip this position unless the lemma window matches the pattern exactly.
            if token_lemmas[i:i + size] != target:
                continue

            matched_text = " ".join(t.text for t in tokens[i:i + size])
            key          = (matched_text.lower(), pattern["phrase"])

            if key in seen:
                continue
            seen.add(key)

            findings.append({
                "phrase":          matched_text,
                "base_phrase":     pattern["phrase"],
                "replacement":     pattern["replacement"],
                "category":        pattern["category"],
                "explanation":     pattern["explanation"],
                "correct_variant": pattern["correct_variant"],
            })

    return findings

# Replaces detected pleonasms with their suggested corrected form.
def apply_pleonasm_replacements(text: str, pleonasms: list[dict]) -> str:
    cleaned = normalize_spacing(text)
    for item in pleonasms:
        phrase      = item["phrase"]
        # Use the first replacement option when alternatives are separated by /.
        replacement = (item.get("replacement", "") or "").split("/")[0].strip()
        if not phrase or not replacement:
            continue
        cleaned = re.sub(
            r"\b" + re.escape(phrase) + r"\b",
            replacement,
            cleaned,
            flags=re.IGNORECASE,
        )
    return normalize_spacing(cleaned)

# ---------------------------------------------------------------------------
# Repeated-word detection
# ---------------------------------------------------------------------------

# Detects repeated semantic lemmas inside each individual sentence.
def find_repeated_words(text: str, nlp) -> list[dict]:
    results = []
    for sentence in split_sentences(text):
        doc    = nlp(sentence)
         # Keep only content-bearing, non-stopword tokens.
        words  = [
            t.lemma_.lower()
            for t in doc
            if t.pos_ in SEMANTIC_POS and not t.is_stop and not t.is_punct
        ]
        # Count lemma occurrences in the current sentence.
        counts     = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        duplicates = [w for w, c in counts.items() if c > 1]
        if duplicates:
            results.append({"sentence": sentence, "words": duplicates})
    return results

# ---------------------------------------------------------------------------
# Similar-word detection  (spaCy word vectors — appropriate at word level)
# ---------------------------------------------------------------------------

# Extracts content tokens suitable for word-level vector similarity.
def _content_tokens(doc, min_len: int = 2):
    return [
        t for t in doc
        if t.pos_ in SEMANTIC_POS
        and not t.is_stop
        and not t.is_punct
        and len(t.text.strip()) > min_len
    ]

# Finds semantically similar word pairs using spaCy lemma vectors.
def find_similar_words(
    doc,
    threshold:  float = 0.75,
    max_tokens: int   = 80,
) -> list[tuple]:
     # Limit token count to keep the pairwise similarity matrix small.
    tokens         = _content_tokens(doc)[:max_tokens]
    vectors, valid = [], []

    for t in tokens:
        # Use the lemma vector so inflected forms map to a common representation.
        v = doc.vocab[t.lemma_].vector
        if v.any():
            vectors.append(v)
            valid.append(t)
    # At least two valid vectors are required for pairwise similarity.
    if len(vectors) < 2:
        return []

    # Build and L2-normalize the vector matrix before cosine similarity.
    mat   = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat  /= np.where(norms == 0, 1e-9, norms)
    sim   = mat @ mat.T

    seen, pairs = set(), []
    rows, cols  = np.where((sim >= threshold) & (sim <= 1.0))

    for i, j in zip(rows, cols):
        if i >= j:
            continue
        t1, t2 = valid[i], valid[j]
        if t1.lemma_.lower() == t2.lemma_.lower():
            continue
        key = frozenset([t1.text.lower(), t2.text.lower()])
        if key in seen:
            continue
        seen.add(key)
        pairs.append((t1.text, t2.text, round(float(sim[i, j]), 2)))

    return sorted(pairs, key=lambda x: -x[2])
# ---------------------------------------------------------------------------
# Redundant-sentence detection  (sentence-transformer cosine similarity)
# ---------------------------------------------------------------------------

# Maps a sentence similarity score to a redundancy label.
def classify_redundancy(score: float) -> str:
    for label, threshold in REDUNDANCY_THRESHOLDS.items():
        if score >= threshold:
            return label
    return "related"

# Finds sentence pairs whose transformer embeddings are above the redundancy threshold.
def find_redundant_sentences(
    sentences: list[str],
    threshold: float = 0.82,
) -> list[tuple]:
    """
    Compares every sentence pair using a multilingual sentence-transformer.

    - No window limit: the full N×N cosine matrix is computed in one batched
      operation, so distant-but-redundant sentences are never missed.
    - No blended weights: transformer cosine alone is accurate enough that
      mixing in weaker signals (Jaccard, WordNet) only adds noise.
    - Threshold raised to 0.82: transformer scores are more compressed than
      raw word-vector averages, so 0.75 would catch too much noise.
    """
    # No comparison is possible with fewer than two sentences.
    if len(sentences) < 2:
        return []

    # Encode all sentences at once — batched GPU/CPU inference
    embeddings = encode_sentences(sentences)          # (N, 768), L2-normalised
    sim_matrix = embeddings @ embeddings.T            # cosine sim == dot product

    # Store unique redundant pairs and avoid duplicate normalized pairs.
    redundant, seen = [], set()

    # Upper triangle only (i < j), at or above threshold
    rows, cols = np.where(
        np.triu(sim_matrix >= threshold, k=1)
    )

    for i, j in zip(rows, cols):
        norm_a = normalize_for_dedup(sentences[i])
        norm_b = normalize_for_dedup(sentences[j])
        key    = frozenset([norm_a, norm_b])

        if key in seen:
            continue
        seen.add(key)

        # Exact duplicates always score 1.0 regardless of transformer output
        score    = 1.0 if norm_a == norm_b else float(sim_matrix[i, j])
        category = classify_redundancy(score)

        redundant.append((
            sentences[i],
            sentences[j],
            round(score, 2),
            category,
        ))

    return sorted(redundant, key=lambda x: -x[2])

# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

# Main public function that runs all text-quality checks and returns one report dictionary.
def analyze_text(
    text:               str,
    word_sim_threshold: float = 0.75,
    sent_sim_threshold: float = 0.82,
    max_similar_tokens: int   = 80,
    nlp                       = None,
) -> dict:
        # Load the large Italian spaCy model only when the caller did not pass one in.
    if nlp is None:
        nlp = spacy.load("it_core_news_lg")
# Reuse the same normalized text, spaCy document, and sentence list across checks.
    text      = normalize_spacing(text)
    doc       = nlp(text)
    sentences = split_sentences(text)

    return {
        "pleonasms":           find_pleonasms(text, nlp),
        "repeated_words":      find_repeated_words(text, nlp),
        "similar_words":       find_similar_words(doc, word_sim_threshold, max_similar_tokens),
        "redundant_sentences": find_redundant_sentences(sentences, sent_sim_threshold),
    }

# ---------------------------------------------------------------------------
# CLI report
# ---------------------------------------------------------------------------

_SEP = "-" * 60



# Prints a report section heading.
def _section(title: str) -> None:
    print(f"\n{_SEP}\n{title}\n{_SEP}")


# Pretty-prints the analysis result returned by analyze_text().
    _section("PLEONASMI")
    if report["pleonasms"]:
        for item in report["pleonasms"]:
            print(f"  ⚠  '{item['phrase']}'")
            print(f"     Forma base:  {item['base_phrase']}")
            print(f"     Correzione:  {item['replacement']}")
            print(f"     Categoria:   {item['category']}")
            if item["explanation"]:
                print(f"     Spiegazione: {item['explanation']}")
            print()
    else:
        print("  Nessun pleonasmo trovato.")

    _section("PAROLE RIPETUTE NELLA STESSA FRASE")
    if report["repeated_words"]:
        for item in report["repeated_words"]:
            print(f"  Parole: {item['words']}")
            print(f"  Frase:  \"{item['sentence']}\"\n")
    else:
        print("  Nessuna ripetizione trovata.")

    _section("PAROLE SIMILI / QUASI SINONIMI")
    if report["similar_words"]:
        for w1, w2, score in report["similar_words"]:
            bar = "█" * int(score * 20)
            print(f"  '{w1}' ↔ '{w2}'  {score:.2f}  {bar}")
    else:
        print("  Nessuna coppia simile trovata.")

    _section("FRASI RIDONDANTI")
    if report["redundant_sentences"]:
        for sent_a, sent_b, score, category in report["redundant_sentences"]:
            print(f"  Similarità: {score:.2f}  [{category}]")
            print(f"  A: {sent_a}")
            print(f"  B: {sent_b}\n")
    else:
        print("  Nessuna frase ridondante trovata.")

    print(f"\n{_SEP}\n")