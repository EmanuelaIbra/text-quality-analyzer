import json
import os
import re
import spacy
import numpy as np
from functools import lru_cache
from nltk.corpus import wordnet as wn


def normalize_spacing(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=[.!?])(?=[A-ZÀ-Ü])", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text.strip()


def split_sentences(text):
    text = normalize_spacing(text)

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def normalize_sentence_for_duplicate(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^\wà-ÿ\s]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()


def load_pleonasm_entries(json_path="data/italian_pleonasms.json"):
    if not os.path.exists(json_path):
        print(f"Warning: pleonasm file not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    entries = []
    categories = data.get("categories", {})

    for category_name, category_data in categories.items():
        for item in category_data.get("examples", []):
            phrase = item.get("pleonasmo", "").strip()
            replacement = item.get("forma_corretta", "").strip()

            if not phrase:
                continue

            entries.append({
                "phrase": phrase,
                "replacement": replacement,
                "category": category_name,
                "explanation": item.get("spiegazione", ""),
                "correct_variant": item.get("variante_corretta", ""),
            })

    return entries


PLEONASM_ENTRIES = load_pleonasm_entries()
_PATTERN_CACHE = {}


def build_lemma_patterns(entries, nlp):
    patterns = []

    for entry in entries:
        phrase_doc = nlp(entry["phrase"])

        lemmas = [
            token.lemma_.lower()
            for token in phrase_doc
            if not token.is_punct
            and not token.is_space
            and token.text.strip()
        ]

        if not lemmas:
            continue

        patterns.append({
            "phrase": entry["phrase"].lower(),
            "replacement": entry["replacement"],
            "category": entry["category"],
            "explanation": entry["explanation"],
            "correct_variant": entry["correct_variant"],
            "lemmas": lemmas,
        })

    return patterns


def get_lemma_patterns(nlp):
    model_name = nlp.meta.get("name", "default")

    if model_name not in _PATTERN_CACHE:
        _PATTERN_CACHE[model_name] = build_lemma_patterns(
            PLEONASM_ENTRIES,
            nlp
        )

    return _PATTERN_CACHE[model_name]


def warmup_pleonasm_cache(nlp):
    get_lemma_patterns(nlp)


def find_pleonasms(text, nlp=None):
    if nlp is None:
        nlp = spacy.load("it_core_news_lg")

    text = normalize_spacing(text)
    doc = nlp(text)

    tokens = [
        token
        for token in doc
        if not token.is_punct
        and not token.is_space
        and token.text.strip()
    ]

    token_lemmas = [
        token.lemma_.lower()
        for token in tokens
    ]

    patterns = get_lemma_patterns(nlp)

    findings = []
    seen = set()

    for pattern in patterns:
        pattern_lemmas = pattern["lemmas"]
        size = len(pattern_lemmas)

        for i in range(len(token_lemmas) - size + 1):
            if token_lemmas[i:i + size] == pattern_lemmas:
                matched_tokens = tokens[i:i + size]
                matched_text = " ".join(
                    token.text for token in matched_tokens
                )

                key = (matched_text.lower(), pattern["phrase"])

                if key in seen:
                    continue

                seen.add(key)

                findings.append({
                    "phrase": matched_text,
                    "base_phrase": pattern["phrase"],
                    "replacement": pattern["replacement"],
                    "category": pattern["category"],
                    "explanation": pattern["explanation"],
                    "correct_variant": pattern["correct_variant"],
                })

    return findings


def clean_replacement_text(replacement):
    if not replacement:
        return ""

    return replacement.split("/")[0].strip()


def apply_pleonasm_replacements(text, pleonasms):
    cleaned = normalize_spacing(text)

    for item in pleonasms:
        phrase = item["phrase"]
        replacement = clean_replacement_text(
            item.get("replacement", "")
        )

        if not phrase or not replacement:
            continue

        pattern = r"\b" + re.escape(phrase) + r"\b"

        cleaned = re.sub(
            pattern,
            replacement,
            cleaned,
            flags=re.IGNORECASE,
        )

    return normalize_spacing(cleaned)


def find_repeated_words(text, nlp):
    results = []
    sentences = split_sentences(text)

    for sentence in sentences:
        doc = nlp(sentence)

        words = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
            and not token.is_stop
            and not token.is_punct
        ]

        counts = {}

        for word in words:
            counts[word] = counts.get(word, 0) + 1

        duplicates = [
            word
            for word, count in counts.items()
            if count > 1
        ]

        if duplicates:
            results.append({
                "sentence": sentence,
                "words": duplicates,
            })

    return results


@lru_cache(maxsize=5000)
def get_wordnet_synset_keys(word):
    try:
        lemmas = wn.lemmas(word, lang="ita")

        keys = set()

        for lemma in lemmas[:5]:
            keys.add(lemma.synset().name())

        return keys

    except Exception:
        return set()


def wordnet_overlap_score(lemmas_a, lemmas_b):
    if not lemmas_a or not lemmas_b:
        return 0.0

    matches = 0
    checked = 0

    for lemma_a in lemmas_a:
        keys_a = get_wordnet_synset_keys(lemma_a)

        if not keys_a:
            continue

        checked += 1

        for lemma_b in lemmas_b:
            keys_b = get_wordnet_synset_keys(lemma_b)

            if keys_a & keys_b:
                matches += 1
                break

    if checked == 0:
        return 0.0

    return matches / checked


def get_generalized_vector(doc):
    semantic_pos = {"NOUN", "VERB", "ADJ", "PROPN", "ADV"}
    clean_vectors = []

    for token in doc:
        if (
            token.pos_ in semantic_pos
            and not token.is_stop
            and not token.is_punct
        ):
            lemma_vector = doc.vocab[token.lemma_].vector

            if lemma_vector.any():
                clean_vectors.append(lemma_vector)

    if not clean_vectors:
        return None

    return np.mean(clean_vectors, axis=0)


def find_similar_words(doc, threshold=0.75, max_tokens=80):
    semantic_pos = {"NOUN", "VERB", "ADJ", "PROPN", "ADV"}

    content_tokens = [
        token
        for token in doc
        if token.pos_ in semantic_pos
        and not token.is_stop
        and not token.is_punct
        and len(token.text) > 2
    ][:max_tokens]

    vectors = []
    valid_tokens = []

    for token in content_tokens:
        vector = doc.vocab[token.lemma_].vector

        if vector.any():
            vectors.append(vector)
            valid_tokens.append(token)

    if len(vectors) < 2:
        return []

    vectors = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    vectors = vectors / norms

    sim_matrix = vectors @ vectors.T

    seen = set()
    pairs = []

    rows, cols = np.where(
        (sim_matrix >= threshold)
        & (sim_matrix <= 1.0)
    )

    for i, j in zip(rows, cols):
        if i >= j:
            continue

        token_1 = valid_tokens[i]
        token_2 = valid_tokens[j]

        if token_1.lemma_.lower() == token_2.lemma_.lower():
            continue

        key = frozenset([
            token_1.text.lower(),
            token_2.text.lower()
        ])

        if key in seen:
            continue

        seen.add(key)

        pairs.append((
            token_1.text,
            token_2.text,
            round(float(sim_matrix[i, j]), 2),
        ))

    return sorted(
        pairs,
        key=lambda item: -item[2]
    )


def get_content_lemmas(doc):
    semantic_pos = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

    return [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in semantic_pos
        and not token.is_stop
        and not token.is_punct
        and len(token.text.strip()) > 2
    ]


def jaccard_similarity(set_a, set_b):
    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def combined_sentence_similarity(doc_a, doc_b):
    vec_a = get_generalized_vector(doc_a)
    vec_b = get_generalized_vector(doc_b)

    if vec_a is not None and vec_b is not None:
        denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)

        if denominator == 0:
            vector_similarity = 0.0
        else:
            vector_similarity = float(
                np.dot(vec_a, vec_b) / denominator
            )
    else:
        vector_similarity = 0.0

    lemmas_a = get_content_lemmas(doc_a)
    lemmas_b = get_content_lemmas(doc_b)

    lemma_overlap = jaccard_similarity(
        set(lemmas_a),
        set(lemmas_b)
    )

    wordnet_overlap = max(
        wordnet_overlap_score(lemmas_a, lemmas_b),
        wordnet_overlap_score(lemmas_b, lemmas_a),
    )

    combined = (
        0.80 * vector_similarity
        + 0.15 * lemma_overlap
        + 0.05 * wordnet_overlap
    )

    if vector_similarity >= 0.75:
        combined = max(combined, vector_similarity)

    return (
        round(vector_similarity, 3),
        round(lemma_overlap, 3),
        round(wordnet_overlap, 3),
        round(min(combined, 1.0), 3),
    )


def classify_redundancy(score):
    if score >= 0.95:
        return "duplicate"

    if score >= 0.80:
        return "manual_review"

    if score >= 0.75:
        return "merge_candidate"

    return "related"


def find_redundant_sentences(
    sentences,
    sent_docs,
    threshold=0.75,
    window=3,
):
    redundant = []
    seen = set()

    for i in range(len(sent_docs)):
        max_j = min(i + window + 1, len(sent_docs))

        for j in range(i + 1, max_j):
            normalized_a = normalize_sentence_for_duplicate(sentences[i])
            normalized_b = normalize_sentence_for_duplicate(sentences[j])

            vector_score, lemma_score, wordnet_score, combined_score = (
                combined_sentence_similarity(
                    sent_docs[i],
                    sent_docs[j],
                )
            )

            if normalized_a == normalized_b:
                final_score = 1.0
                category = "duplicate"
            else:
                final_score = combined_score
                category = classify_redundancy(final_score)

            if final_score >= threshold:
                key = frozenset([
                    normalized_a,
                    normalized_b,
                ])

                if key in seen:
                    continue

                seen.add(key)

                redundant.append((
                    sentences[i],
                    sentences[j],
                    round(final_score, 2),
                    category,
                    {
                        "vector_similarity": vector_score,
                        "lemma_overlap": lemma_score,
                        "wordnet_overlap": wordnet_score,
                    },
                ))

    return sorted(
        redundant,
        key=lambda item: -item[2]
    )


def analyze_text(
    text,
    word_sim_threshold=0.75,
    sent_sim_threshold=0.75,
    max_similar_tokens=80,
    sentence_window=3,
    nlp=None,
):
    if nlp is None:
        nlp = spacy.load("it_core_news_lg")

    text = normalize_spacing(text)
    doc = nlp(text)
    sentences = split_sentences(text)
    sent_docs = list(nlp.pipe(sentences))

    return {
        "pleonasms": find_pleonasms(text, nlp),
        "repeated_words": find_repeated_words(text, nlp),
        "similar_words": find_similar_words(
            doc,
            threshold=word_sim_threshold,
            max_tokens=max_similar_tokens,
        ),
        "redundant_sentences": find_redundant_sentences(
            sentences,
            sent_docs,
            threshold=sent_sim_threshold,
            window=sentence_window,
        ),
        "synonyms": {},
    }


def print_report(report):
    sep = "-" * 60

    print("\n" + sep)
    print("PLEONASMI")
    print(sep)

    if report["pleonasms"]:
        for item in report["pleonasms"]:
            print(f"  ⚠  '{item['phrase']}'")
            print(f"     Forma base: {item['base_phrase']}")
            print(f"     Correzione: {item['replacement']}")
            print(f"     Categoria: {item['category']}")

            if item["explanation"]:
                print(f"     Spiegazione: {item['explanation']}")

            print()
    else:
        print("  Nessun pleonasmo trovato.")

    print("\n" + sep)
    print("PAROLE RIPETUTE NELLA STESSA FRASE")
    print(sep)

    if report["repeated_words"]:
        for item in report["repeated_words"]:
            print(f"  Parole: {item['words']}")
            print(f"  Frase:  \"{item['sentence']}\"\n")
    else:
        print("  Nessuna ripetizione trovata.")

    print("\n" + sep)
    print("PAROLE SIMILI / QUASI SINONIMI")
    print(sep)

    if report["similar_words"]:
        for word_1, word_2, score in report["similar_words"]:
            bar = "█" * int(score * 20)
            print(f"  '{word_1}' ↔ '{word_2}'  {score:.2f}  {bar}")
    else:
        print("  Nessuna coppia simile trovata.")

    print("\n" + sep)
    print("FRASI RIDONDANTI")
    print(sep)

    if report["redundant_sentences"]:
        for item in report["redundant_sentences"]:
            print(f"  Similarità combinata totale: {item[2]:.2f}")
            print(f"  Categoria: {item[3]}")
            print(f"  Vector Semantics Lemmas: {item[4]['vector_similarity']}")
            print(f"  Lemma Overlap: {item[4]['lemma_overlap']}")
            print(f"  WordNet Overlap: {item[4]['wordnet_overlap']}")
            print(f"  A: {item[0]}")
            print(f"  B: {item[1]}\n")
    else:
        print("  Nessuna frase ridondante trovata.")

    print("\n" + sep + "\n")