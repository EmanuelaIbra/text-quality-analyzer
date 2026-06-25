import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

SEMANTIC_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

# Sentence-transformer cosine scores are conservative. The merge threshold is
# intentionally lower than the manual-review threshold so related Italian
# sentences are shown to the user instead of silently ignored.
REDUNDANCY_THRESHOLDS = {
    "duplicate": 0.95,
    "manual_review": 0.85,
    "merge_candidate": 0.70,
}

SENTENCE_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_PLEONASM_PATH = DATA_DIR / "italian_pleonasms.json"


# ---------------------------------------------------------------------------
# Sentence-transformer model
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_sentence_model():
    """Load the multilingual sentence model once and reuse it for all calls."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(SENTENCE_MODEL_NAME, device="cpu")


def encode_sentences(sentences: list[str]) -> np.ndarray:
    """Batch-encode sentences and return L2-normalised vectors."""
    return get_sentence_model().encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )


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
# Pleonasm loading and detection
# ---------------------------------------------------------------------------

def load_pleonasm_entries(json_path: str | Path | None = None) -> list[dict]:
    """Load Italian pleonasm examples from the project data file."""
    path = Path(json_path) if json_path else DEFAULT_PLEONASM_PATH
    if not path.exists():
        print(f"Warning: pleonasm file not found: {path}")
        return []

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for category_name, category_data in data.get("categories", {}).items():
        for item in category_data.get("examples", []):
            phrase = item.get("pleonasmo", "").strip()
            if not phrase:
                continue
            entries.append(
                {
                    "phrase": phrase,
                    "replacement": item.get("forma_corretta", "").strip(),
                    "category": category_name,
                    "explanation": item.get("spiegazione", ""),
                    "correct_variant": item.get("variante_corretta", ""),
                }
            )
    return entries


# Load pleonasm entries once at module import time.
PLEONASM_ENTRIES = load_pleonasm_entries()
# Cache lemma-based patterns per spaCy model name to avoid rebuilding them repeatedly.
_PATTERN_CACHE: dict[tuple[str, str], list[dict]] = {}

# Converts pleonasm phrases into lemma sequences for inflection-tolerant matching.
def _build_lemma_patterns(entries: list[dict], nlp) -> list[dict]:
    """Convert pleonasm phrases into lemma patterns for flexible matching."""
    patterns = []
    for entry in entries:
        lemmas = [
            token.lemma_.lower()
            for token in nlp(entry["phrase"])
            if not token.is_punct and not token.is_space and token.text.strip()
        ]
        if lemmas:
            patterns.append({**entry, "phrase": entry["phrase"].lower(), "lemmas": lemmas})
    return patterns



# Retrieves cached lemma patterns for the current spaCy pipeline.
def get_lemma_patterns(nlp) -> list[dict]:
    """Return cached pleonasm lemma patterns for the active spaCy pipeline."""
    key = (
        nlp.meta.get("name", "default"),
        nlp.meta.get("version", "unknown"),
    )
    if key not in _PATTERN_CACHE:
        _PATTERN_CACHE[key] = _build_lemma_patterns(PLEONASM_ENTRIES, nlp)
    return _PATTERN_CACHE[key]


def warmup_pleonasm_cache(nlp) -> None:
    """Prebuild pleonasm patterns during application startup."""
    get_lemma_patterns(nlp)


def find_pleonasms(text: str, nlp) -> list[dict]:
    """Find pleonastic expressions by comparing token lemmas."""
    doc = nlp(normalize_spacing(text))
    tokens = [t for t in doc if not t.is_punct and not t.is_space and t.text.strip()]
    token_lemmas = [t.lemma_.lower() for t in tokens]
    findings, seen = [], set()

    for pattern in get_lemma_patterns(nlp):
        size = len(pattern["lemmas"])
        target = pattern["lemmas"]

        for index in range(len(token_lemmas) - size + 1):
            if token_lemmas[index:index + size] != target:
                continue

            matched_text = " ".join(t.text for t in tokens[index:index + size])
            key = (matched_text.lower(), pattern["phrase"])
            if key in seen:
                continue
            seen.add(key)

            findings.append(
                {
                    "phrase": matched_text,
                    "base_phrase": pattern["phrase"],
                    "replacement": pattern["replacement"],
                    "category": pattern["category"],
                    "explanation": pattern["explanation"],
                    "correct_variant": pattern["correct_variant"],
                }
            )

    return findings


def apply_pleonasm_replacements(text: str, pleonasms: list[dict]) -> str:
    """Replace detected pleonasms with their suggested corrected form."""
    cleaned = normalize_spacing(text)
    for item in pleonasms:
        phrase = item.get("phrase", "")
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
# Repeated and similar words
# ---------------------------------------------------------------------------

def _content_tokens(doc, min_len: int = 2):
    """Extract content-bearing tokens suitable for repetition checks."""
    return [
        token
        for token in doc
        if token.pos_ in SEMANTIC_POS
        and not token.is_stop
        and not token.is_punct
        and len(token.text.strip()) > min_len
    ]


def find_repeated_words(text: str, nlp, doc=None) -> list[dict]:
    """Detect repeated semantic lemmas inside each sentence."""
    doc = doc or nlp(normalize_spacing(text))
    spans = list(doc.sents) if doc.has_annotation("SENT_START") else []
    sentence_docs = spans or [nlp(sentence) for sentence in split_sentences(text)]

    results = []
    for sentence_doc in sentence_docs:
        counts: dict[str, int] = {}
        for token in _content_tokens(sentence_doc):
            lemma = token.lemma_.lower()
            counts[lemma] = counts.get(lemma, 0) + 1

        duplicates = [lemma for lemma, count in counts.items() if count > 1]
        if duplicates:
            results.append({"sentence": sentence_doc.text.strip(), "words": duplicates})
    return results


def find_similar_words(
    doc,
    threshold: float = 0.75,
    max_tokens: int = 80,
) -> list[tuple]:
    """Find semantically similar word pairs using spaCy lemma vectors."""
    tokens = _content_tokens(doc)[:max_tokens]
    vectors, valid_tokens = [], []

    for token in tokens:
        vector = doc.vocab[token.lemma_].vector
        if vector.any():
            vectors.append(vector)
            valid_tokens.append(token)

    if len(vectors) < 2:
        return []

    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.where(norms == 0, 1e-9, norms)
    similarity = matrix @ matrix.T

    seen, pairs = set(), []
    rows, cols = np.where((similarity >= threshold) & (similarity < 1.0))

    for i, j in zip(rows, cols):
        if i >= j:
            continue
        first, second = valid_tokens[i], valid_tokens[j]
        if first.lemma_.lower() == second.lemma_.lower():
            continue
        key = frozenset([first.text.lower(), second.text.lower()])
        if key in seen:
            continue
        seen.add(key)
        pairs.append((first.text, second.text, round(float(similarity[i, j]), 2)))

    return sorted(pairs, key=lambda item: -item[2])


# ---------------------------------------------------------------------------
# Redundant sentence detection
# ---------------------------------------------------------------------------

def _sentence_lemmas(sentence: str, nlp=None) -> set[str]:
    """Return content lemmas for lexical backup scoring."""
    if nlp is not None:
        return {
            token.lemma_.lower()
            for token in nlp(sentence)
            if token.pos_ in SEMANTIC_POS and not token.is_stop and not token.is_punct
        }
    return {
        token
        for token in re.findall(r"[a-zà-ÿ]+", sentence.lower())
        if len(token) > 3
    }


def _lexical_redundancy_score(first: set[str], second: set[str]) -> float:
    """
    Backup score for near-duplicate Italian sentences.

    Transformer similarity is the main signal. This lexical score catches
    short, highly overlapping sentences that multilingual embeddings sometimes
    score too conservatively.
    """
    if len(first) < 3 or len(second) < 3:
        return 0.0

    overlap = len(first & second)
    if overlap == 0:
        return 0.0

    containment = overlap / min(len(first), len(second))
    jaccard = overlap / len(first | second)

    if containment >= 0.80 and jaccard >= 0.50:
        return min(0.87, 0.74 + (0.16 * jaccard))
    if containment >= 0.65 and jaccard >= 0.45:
        return min(0.78, 0.68 + (0.14 * jaccard))
    return jaccard


def classify_redundancy(score: float) -> str:
    """Map a sentence similarity score to a redundancy label."""
    for label, threshold in REDUNDANCY_THRESHOLDS.items():
        if score >= threshold:
            return label
    return "related"


def find_redundant_sentences(
    sentences: list[str],
    threshold: float = 0.80,
    nlp=None,
) -> list[tuple]:
    """
    Find redundant or mergeable sentence pairs.

    The main signal is multilingual transformer cosine similarity. Exact
    duplicates are detected before model inference, and a conservative lexical
    backup helps with short Italian sentences that share most content words.
    """
    if len(sentences) < 2:
        return []

    normalised = [normalize_for_dedup(sentence) for sentence in sentences]
    lemma_sets = [_sentence_lemmas(sentence, nlp) for sentence in sentences]
    redundant, seen = [], set()
    non_exact_pairs = []

    # Exact duplicates are cheap and deterministic; catch them before loading
    # the transformer model.
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            key = frozenset([normalised[i], normalised[j]])
            if key in seen:
                continue
            seen.add(key)

            if normalised[i] == normalised[j]:
                redundant.append(
                    (
                        sentences[i],
                        sentences[j],
                        1.0,
                        classify_redundancy(1.0),
                    )
                )
            else:
                non_exact_pairs.append((i, j))

    if not non_exact_pairs:
        return sorted(redundant, key=lambda item: -item[2])

    transformer_scores = None

    try:
        embeddings = encode_sentences(sentences)
        transformer_scores = embeddings @ embeddings.T
    except Exception as exc:
        print(f"Warning: sentence transformer unavailable, using lexical fallback: {exc}")

    for i, j in non_exact_pairs:
        transformer_score = (
            float(transformer_scores[i, j])
            if transformer_scores is not None
            else 0.0
        )
        lexical_score = _lexical_redundancy_score(lemma_sets[i], lemma_sets[j])
        score = max(transformer_score, lexical_score)

        if score < threshold:
            continue

        redundant.append(
            (
                sentences[i],
                sentences[j],
                round(score, 2),
                classify_redundancy(score),
            )
        )

    return sorted(redundant, key=lambda item: -item[2])


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def analyze_text(
    text: str,
    word_sim_threshold: float = 0.82,
    sent_sim_threshold: float = 0.72,
    max_similar_tokens: int = 80,
    nlp=None,
) -> dict:
    """Run all text-quality checks and return one report dictionary."""
    if not text or not text.strip():
        return {
            "pleonasms": [],
            "repeated_words": [],
            "similar_words": [],
            "redundant_sentences": [],
        }

    if nlp is None:
        import spacy

        nlp = spacy.load("it_core_news_lg")

    text = normalize_spacing(text)
    doc = nlp(text)
    sentences = [span.text.strip() for span in doc.sents] or split_sentences(text)

    return {
        "pleonasms": find_pleonasms(text, nlp),
        "repeated_words": find_repeated_words(text, nlp, doc=doc),
        "similar_words": find_similar_words(doc, word_sim_threshold, max_similar_tokens),
        "redundant_sentences": find_redundant_sentences(
            sentences,
            threshold=sent_sim_threshold,
            nlp=nlp,
        ),
    }


# ---------------------------------------------------------------------------
# CLI report helper
# ---------------------------------------------------------------------------

_SEP = "-" * 60


def _section(title: str) -> None:
    print(f"\n{_SEP}\n{title}\n{_SEP}")


def print_report(report: dict) -> None:
    """Pretty-print an analyze_text() report for manual debugging."""
    _section("PLEONASMI")
    if report["pleonasms"]:
        for item in report["pleonasms"]:
            print(f"  - '{item['phrase']}'")
            print(f"    Forma base:  {item['base_phrase']}")
            print(f"    Correzione:  {item['replacement']}")
            print(f"    Categoria:   {item['category']}")
            if item["explanation"]:
                print(f"    Spiegazione: {item['explanation']}")
    else:
        print("  Nessun pleonasmo trovato.")

    _section("PAROLE RIPETUTE NELLA STESSA FRASE")
    if report["repeated_words"]:
        for item in report["repeated_words"]:
            print(f"  Parole: {item['words']}")
            print(f"  Frase:  \"{item['sentence']}\"")
    else:
        print("  Nessuna ripetizione trovata.")

    _section("PAROLE SIMILI / QUASI SINONIMI")
    if report["similar_words"]:
        for first, second, score in report["similar_words"]:
            print(f"  '{first}' <-> '{second}'  {score:.2f}")
    else:
        print("  Nessuna coppia simile trovata.")

    _section("FRASI RIDONDANTI")
    if report["redundant_sentences"]:
        for first, second, score, category in report["redundant_sentences"]:
            print(f"  Similarita: {score:.2f}  [{category}]")
            print(f"  A: {first}")
            print(f"  B: {second}")
    else:
        print("  Nessuna frase ridondante trovata.")

    print(f"\n{_SEP}\n")
