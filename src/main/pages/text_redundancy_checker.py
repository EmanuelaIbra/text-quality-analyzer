import json
import os
import re
import spacy


def normalize_spacing(text):
    """
    Cleans up text spacing by replacing multiple whitespaces with a single space 
    and ensuring a space exists between adjacent sentences.
    """
    # Replace any sequence of whitespace characters (tabs, newlines) with a single space
    text = re.sub(r"\s+", " ", text)
    # Ensure there's a space if a capital letter/accented letter immediately follows sentence punctuation
    text = re.sub(r"(?<=[.!?])(?=[A-ZÀ-Ü])", " ", text)
    return text.strip()


def split_sentences(text):
    """
    Normalizes spacing and splits the text into a list of clean, non-empty sentences 
    using punctuation marks (. ! ?) as boundaries.
    """
    text = normalize_spacing(text)

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def load_pleonasm_entries(json_path="data/italian_pleonasms.json"):
    """
    Loads and flattens a JSON structure containing known Italian pleonasms 
    (redundant expressions like "a me mi") categorized by type.
    """
    # Safe check to verify if the dictionary file exists before opening it
    if not os.path.exists(json_path):
        print(f"Warning: pleonasm file not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    entries = []
    categories = data.get("categories", {})

    # Extract items sequentially out of nested JSON categories
    for category_name, category_data in categories.items():
        examples = category_data.get("examples", [])

        for item in examples:
            phrase = item.get("pleonasmo", "").strip()
            replacement = item.get("forma_corretta", "").strip()

            # Skip empty entries if they exist in the JSON
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

# Globally load and initialize the list of pleonasms on script execution
PLEONASM_ENTRIES = load_pleonasm_entries()

_PATTERN_CACHE = {}

def build_lemma_patterns(entries, nlp):
    """
    Converts pleonasm phrases into lists of lowercase dictionary base forms (lemmas) 
    to enable flexible matching across word inflections.
    """
    patterns = []

    for entry in entries:
        phrase = entry["phrase"]
        replacement = entry["replacement"]

        # Run the lookup expression through spaCy to find its base lemmas
        phrase_doc = nlp(phrase)

        lemmas = [
            token.lemma_.lower()
            for token in phrase_doc
            if (
                not token.is_punct
                and not token.is_space
                and token.text.strip()
            )
        ]

        if not lemmas:
            continue

        patterns.append({
            "phrase": phrase.lower(),
            "replacement": replacement,
            "category": entry["category"],
            "explanation": entry["explanation"],
            "correct_variant": entry["correct_variant"],
            "lemmas": lemmas,
        })

    return patterns

def get_lemma_patterns(nlp):
    """
    Builds pleonasm lemma patterns once per spaCy model and reuses them.
    This avoids reprocessing the JSON pleonasm list on every request.
    """
    global _PATTERN_CACHE

    model_name = nlp.meta.get("name", "default")

    if model_name not in _PATTERN_CACHE:
        _PATTERN_CACHE[model_name] = build_lemma_patterns(
            PLEONASM_ENTRIES,
            nlp
        )

    return _PATTERN_CACHE[model_name]

def find_pleonasms(text, nlp=None):
    """
    Scans text tokens using a sliding window algorithm to detect matching 
    sequences of lemma patterns linked to defined pleonasms.
    """
    if nlp is None:
        nlp = spacy.load("it_core_news_lg")

    text = normalize_spacing(text)
    doc = nlp(text)

    # Isolate textual content tokens, filtering out whitespace or punctuation marks
    tokens = [
        token
        for token in doc
        if (
            not token.is_punct
            and not token.is_space
            and token.text.strip()
        )
    ]

    # Convert document tokens to lowercase base forms
    token_lemmas = [
        token.lemma_.lower()
        for token in tokens
    ]

    patterns = get_lemma_patterns(nlp)

    findings = []
    seen = set()

    # Slide a variable-width window over text lemmas to match pattern lengths
    for pattern in patterns:
        pattern_lemmas = pattern["lemmas"]
        size = len(pattern_lemmas)

        if size == 0:
            continue

        # Check every possible slice of the text matching the current pattern size    
        for i in range(len(token_lemmas) - size + 1):
            window = token_lemmas[i:i + size]

            if window == pattern_lemmas:
                matched_tokens = tokens[i:i + size]
                matched_text = " ".join(
                    token.text for token in matched_tokens
                )

                key = (matched_text.lower(), pattern["phrase"])


                # Avoid reporting identical overlapping match duplicates
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
    """
    Helper function to split option groups in correction lists 
    (e.g., matching "forma_a / forma_b" options and keeping only the first one).
    """
    if not replacement:
        return ""

    replacement = replacement.split("/")[0].strip()
    return replacement


def apply_pleonasm_replacements(text, pleonasms):
    """
    Iterates through detected pleonasm structures and directly replaces 
    the exact phrase occurrences within the text.
    """
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
    """
    Scans individual sentences for content-heavy tokens (nouns, verbs, adjectives) 
    that appear more than once within the boundaries of that single sentence.
    """
    results = []
    sentences = split_sentences(text)

    for sentence in sentences:
        doc = nlp(sentence)

        # Retain only core lexical terms, skipping common stop words or markers
        words = [
            token.lemma_.lower()
            for token in doc
            if (
                token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
                and not token.is_stop
                and not token.is_punct
            )
        ]

        counts = {}

        # Count frequencies manually within the local sentence
        for word in words:
            counts[word] = counts.get(word, 0) + 1

        # Isolate entries with counts exceeding 1
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


def find_similar_words(doc, threshold=0.88):
    """
    Compares vector representations of all main lexical content tokens across 
    the document to flag different words with dangerously high semantic similarity.
    """
    # Define minor structural word classes to completely ignore during comparison
    skip_pos = {
        "DET", "ADP", "PUNCT", "PRON", "PART", "AUX",
        "CCONJ", "SCONJ", "SPACE", "NUM"
    }

    content_tokens = [
        token
        for token in doc
        if (
            token.pos_ not in skip_pos
            and token.has_vector
            and not token.is_stop
            and len(token.text) > 2
        )
    ]

    pairs = []

    # Run nested index loops to check every word combination pair unique combinations
    for i, token_1 in enumerate(content_tokens):
        for token_2 in content_tokens[i + 1:]:
            # If the base lemmas match exactly, they are variations of the same word (handled elsewhere)
            if token_1.lemma_.lower() == token_2.lemma_.lower():
                continue

            # Calculate cosine similarity using spaCy's embedding vectors
            similarity = float(token_1.similarity(token_2))

            # Flag if the similarity hits the target floor value but is not identical
            if threshold <= similarity < 1.0:
                pairs.append((
                    token_1.text,
                    token_2.text,
                    round(similarity, 2),
                ))

    seen = set()
    unique = []

    # Sort results showing highest similarities first, deduping order pairings (A-B vs B-A)
    for word_1, word_2, score in sorted(
        pairs,
        key=lambda item: -item[2]
    ):
        key = frozenset([word_1.lower(), word_2.lower()])

        if key not in seen:
            seen.add(key)
            unique.append((word_1, word_2, score))

    return unique


def find_redundant_sentences(sentences, sent_docs, threshold=0.88):
    """
    Compares complete sentence embeddings across the entire document list 
    to flag full statements that repeat the same conceptual idea.
    """
    redundant = []

    # Compare sentence combinations without repeating matching positions
    for i in range(len(sent_docs)):
        for j in range(i + 1, len(sent_docs)):
            similarity = float(sent_docs[i].similarity(sent_docs[j]))

            if threshold <= similarity < 1.0:
                redundant.append((
                    sentences[i],
                    sentences[j],
                    round(similarity, 2),
                ))

    return sorted(redundant, key=lambda item: -item[2])


def analyze_text(
    text,
    word_sim_threshold=0.88,
    sent_sim_threshold=0.88,
    nlp=None,
):  
    """
    Main aggregator entry point. Parses sentences and initializes spaCy tokens 
    to generate a complete report mapping redundancies, pleonasms, and styling errors.
    """
    if nlp is None:
        nlp = spacy.load("it_core_news_lg")

    text = normalize_spacing(text)
    doc = nlp(text)
    sentences = split_sentences(text)
    sent_docs = [nlp(sentence) for sentence in sentences]

    return {
        "pleonasms": find_pleonasms(text, nlp),
        "repeated_words": find_repeated_words(text, nlp),
        "similar_words": find_similar_words(
            doc,
            threshold=word_sim_threshold,
        ),
        "redundant_sentences": find_redundant_sentences(
            sentences,
            sent_docs,
            threshold=sent_sim_threshold,
        ),
        "synonyms": {},
    }


def print_report(report):
    """
    Prints a cleanly formatted command line diagnostic report 
    with visual indicator bars measuring similarity outputs.
    """
    sep = "-" * 60

    # SECTION 1: PLEONASMS
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

    # SECTION 2: REPEATED WORDS
    print("\n" + sep)
    print("PAROLE RIPETUTE NELLA STESSA FRASE")
    print(sep)

    if report["repeated_words"]:
        for item in report["repeated_words"]:
            print(f"  Parole: {item['words']}")
            print(f"  Frase:  \"{item['sentence']}\"\n")
    else:
        print("  Nessuna ripetizione trovata.")

    # SECTION 3: SIMILAR WORDS / SEMANTIC NEAR-SYNONYMS
    print("\n" + sep)
    print("PAROLE SIMILI / QUASI SINONIMI")
    print(sep)

    if report["similar_words"]:
        for word_1, word_2, score in report["similar_words"]:
            bar = "█" * int(score * 20)
            print(f"  '{word_1}' ↔ '{word_2}'  {score:.2f}  {bar}")
    else:
        print("  Nessuna coppia simile trovata.")

    # SECTION 4: REDUNDANT SENTENCES
    print("\n" + sep)
    print("FRASI RIDONDANTI")
    print(sep)

    if report["redundant_sentences"]:
        for sent_1, sent_2, score in report["redundant_sentences"]:
            print(f"  Similarità: {score:.2f}")
            print(f"  A: {sent_1}")
            print(f"  B: {sent_2}\n")
    else:
        print("  Nessuna frase ridondante trovata.")

    print("\n" + sep + "\n")