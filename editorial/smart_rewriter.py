"""
editorial/smart_rewriter.py
----------------------------
Smart text improvement engine. Instead of a manual list of fixes,
this module uses NLP logic to detect and fix problems automatically:

  1. Pleonasm detector  — finds redundant word pairs ("global international",
                          "cash money", "huge massive") by comparing the
                          embeddings of adjacent modifiers/nouns.

  2. Synonym replacer   — replaces overused words with contextually
                          appropriate synonyms using WordNet + spaCy POS.

  3. Double negative    — detects and rewrites double negatives.

  4. Tautology checker  — finds sentences that repeat the same idea twice
                          in different words within a single sentence.

  5. Redundant hedges   — removes stacked hedges ("In my personal opinion,
                          I think that...").

  6. Agreement fixer    — catches basic subject-verb and pronoun agreement
                          errors that the GEC model might miss.
"""

import re
import random
import spacy
import nltk
from sentence_transformers import SentenceTransformer, util

try:
    nltk.download("wordnet",   quiet=True)
    nltk.download("omw-1.4",   quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
except Exception:
    pass

from nltk.corpus import wordnet


# ---------------------------------------------------------------------------
# Shared model handles (loaded once)
# ---------------------------------------------------------------------------

_nlp         = None
_embed_model = None


def _ensure_models():
    global _nlp, _embed_model
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        _embed_model.encode(["warmup"], show_progress_bar=False)


# ---------------------------------------------------------------------------
# 1. Pleonasm detector
#    Finds pairs of adjacent words (within 3 tokens) that mean the same thing.
#    "global international" / "cash money" / "huge massive" / "end result"
# ---------------------------------------------------------------------------

def detect_pleonasms(text: str, sim_threshold: float = 0.72) -> list[dict]:
    """
    Automatically detect redundant word pairs in text using embeddings.
    No hardcoded list — works on any word combination.

    Strategy:
      - Look at all (modifier, head) pairs and (adj, adj) pairs within
        a window of 3 tokens
      - Embed each word individually
      - If cosine similarity > threshold they are near-synonyms → pleonasm

    Returns list of dicts: {word_a, word_b, similarity, context, suggestion}
    """
    _ensure_models()
    doc    = _nlp(text)
    found  = []
    tokens = [t for t in doc if not t.is_punct and not t.is_space]

    # Pairs to check: (ADJ, ADJ), (ADJ, NOUN where noun ≈ adj meaning),
    #                 (NOUN, NOUN compounds), (ADV, ADJ)
    check_pos_pairs = {
        ("ADJ",  "ADJ"),
        ("ADV",  "ADJ"),
        ("NOUN", "NOUN"),
        ("ADJ",  "NOUN"),
        ("VERB", "NOUN"),
    }

    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 4, len(tokens))):
            t_a = tokens[i]
            t_b = tokens[j]
            pair = (t_a.pos_, t_b.pos_)
            if pair not in check_pos_pairs:
                continue
            # Skip stop words and short words
            if t_a.is_stop or t_b.is_stop:
                continue
            if len(t_a.lemma_) < 3 or len(t_b.lemma_) < 3:
                continue
            # Skip identical lemmas (caught by repetition module)
            if t_a.lemma_.lower() == t_b.lemma_.lower():
                continue

            embs = _embed_model.encode(
                [t_a.lemma_.lower(), t_b.lemma_.lower()],
                show_progress_bar=False
            )
            sim = float(util.cos_sim(embs[0], embs[1]))

            if sim >= sim_threshold:
                # Decide which word to keep (prefer the more common / shorter)
                keep    = t_a.text if len(t_a.text) <= len(t_b.text) else t_b.text
                remove  = t_b.text if keep == t_a.text else t_a.text
                context = text[max(0, t_a.idx - 20): t_b.idx + len(t_b.text) + 20]

                found.append({
                    "word_a":     t_a.text,
                    "word_b":     t_b.text,
                    "lemma_a":    t_a.lemma_.lower(),
                    "lemma_b":    t_b.lemma_.lower(),
                    "pos_pair":   f"{t_a.pos_}/{t_b.pos_}",
                    "similarity": round(sim, 3),
                    "keep":       keep,
                    "remove":     remove,
                    "context":    context.strip(),
                    "type":       "pleonasm",
                })

    # Deduplicate (same pair found twice)
    seen   = set()
    unique = []
    for p in found:
        key = tuple(sorted([p["lemma_a"], p["lemma_b"]]))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


def apply_pleonasm_fixes(text: str, pleonasms: list[dict]) -> tuple[str, list[dict]]:
    """
    Remove the redundant word from each detected pleonasm.
    Returns (fixed_text, list_of_changes).
    """
    result  = text
    changes = []

    # Sort by position (rightmost first so indices don't shift)
    sorted_p = sorted(pleonasms, key=lambda p: text.find(p["remove"]), reverse=True)

    for p in sorted_p:
        remove_word = p["remove"]
        # Build a pattern that removes the word + surrounding space
        pattern = re.compile(
            r'\b' + re.escape(remove_word) + r'\b\s*',
            flags=re.IGNORECASE
        )
        new_text = pattern.sub("", result, count=1)
        if new_text != result:
            changes.append({
                "original":    remove_word,
                "replacement": "(removed — synonym of '" + p["keep"] + "')",
                "similarity":  p["similarity"],
            })
            result = new_text

    # Clean up double spaces
    result = re.sub(r" {2,}", " ", result).strip()
    return result, changes


# ---------------------------------------------------------------------------
# 2. Synonym replacer for overused words
# ---------------------------------------------------------------------------

# POS tag map: spaCy → WordNet
_POS_MAP = {
    "NOUN": wordnet.NOUN,
    "VERB": wordnet.VERB,
    "ADJ":  wordnet.ADJ,
    "ADV":  wordnet.ADV,
}

# Words we never replace (structural/core vocabulary)
_NEVER_REPLACE = {
    "be","have","do","say","get","make","go","know","take","see","come",
    "want","look","use","find","give","think","tell","become","show",
    "the","a","an","and","or","but","in","on","at","to","for","of",
    "with","by","from","that","this","it","he","she","they","we","i",
    "not","very","also","more","most","much","many","some","any","all",
    "bank","banking","money","world","economy",  # domain words — keep as-is
}


def get_synonyms(word: str, pos: str, n: int = 5) -> list[str]:
    """
    Get WordNet synonyms for a word given its spaCy POS tag.
    Returns up to n synonyms, filtered for common words only.
    """
    wn_pos = _POS_MAP.get(pos)
    if wn_pos is None:
        return []

    synsets   = wordnet.synsets(word.lower(), pos=wn_pos)
    synonyms  = set()
    for syn in synsets[:3]:                    # top 3 synsets only
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if (
                name.lower() != word.lower()
                and "_" not in name            # no multi-word
                and len(name) > 2
                and name.lower() not in _NEVER_REPLACE
            ):
                synonyms.add(name.lower())

    return list(synonyms)[:n]


def replace_overused_words(
    text:         str,
    overused:     list[dict],   # from repetition.find_overused_words()
    min_count:    int = 3,      # only replace if used >= this many times
    max_replaces: int = 2,      # replace at most this many occurrences per word
) -> tuple[str, list[dict]]:
    """
    Replace some occurrences of overused words with contextually appropriate synonyms.

    Strategy:
      - Keep the FIRST occurrence (introduce the word)
      - Replace subsequent occurrences with synonyms
      - Use spaCy POS to find the right WordNet synset
      - Pick the synonym that is most contextually different (max diversity)

    Returns (fixed_text, list_of_changes)
    """
    _ensure_models()
    doc     = _nlp(text)
    result  = text
    changes = []

    for entry in overused:
        lemma = entry["lemma"]
        count = entry["count"]

        if count < min_count or lemma in _NEVER_REPLACE:
            continue

        # Find all tokens matching this lemma
        matching_tokens = [
            t for t in doc
            if t.lemma_.lower() == lemma and not t.is_stop
        ]
        if len(matching_tokens) < 2:
            continue

        # Get POS of first match
        pos      = matching_tokens[0].pos_
        synonyms = get_synonyms(lemma, pos)
        if not synonyms:
            continue

        # Skip the first occurrence, replace up to max_replaces of the rest
        to_replace = matching_tokens[1 : 1 + max_replaces]

        for i, token in enumerate(to_replace):
            synonym = synonyms[i % len(synonyms)]

            # Preserve original capitalisation
            original_form = token.text
            if original_form[0].isupper():
                synonym = synonym[0].upper() + synonym[1:]

            # Replace in text (use character offsets for precision)
            result = (
                result[:token.idx] +
                synonym +
                result[token.idx + len(token.text):]
            )
            changes.append({
                "original":    original_form,
                "replacement": synonym,
                "lemma":       lemma,
                "reason":      f"overused ({count}×) — replaced with synonym",
            })

            # Re-parse after replacement (offsets change)
            doc = _nlp(result)
            matching_tokens = [
                t for t in doc
                if t.lemma_.lower() == lemma and not t.is_stop
            ]

    return result, changes


# ---------------------------------------------------------------------------
# 3. Redundant hedge detector & remover
#    "In my personal opinion, I think that" → "I think"
#    "It is worth noting that" → (remove)
#    "As a matter of fact" → (remove)
# ---------------------------------------------------------------------------

# These are ordered longest-first so longer patterns match before subsets
HEDGE_PATTERNS = [
    (r"in\s+my\s+(?:personal\s+)?opinion\s*,?\s*i\s+(?:personally\s+)?think\s+that\s*",  "I think "),
    (r"in\s+my\s+(?:personal\s+)?opinion\s*,?\s*i\s+(?:personally\s+)?believe\s+that\s*","I believe "),
    (r"in\s+my\s+(?:personal\s+)?opinion\s*,?\s*",                                        ""),
    (r"i\s+personally\s+think\s+that\s*",                                                 "I think "),
    (r"i\s+personally\s+believe\s+that\s*",                                               "I believe "),
    (r"it\s+is\s+(?:important|worth)\s+(?:to\s+)?note\s+that\s*",                        ""),
    (r"it\s+should\s+be\s+noted\s+that\s*",                                               ""),
    (r"it\s+goes\s+without\s+saying\s+that\s*",                                           ""),
    (r"as\s+a\s+matter\s+of\s+fact\s*,?\s*",                                              ""),
    (r"needless\s+to\s+say\s*,?\s*",                                                      ""),
    (r"of\s+course\s*,?\s*",                                                               ""),
    (r"obviously\s*,?\s*",                                                                 ""),
    (r"basically\s*,?\s*",                                                                 ""),
    (r"essentially\s*,?\s*",                                                                ""),
    (r"generally\s+speaking\s*,?\s*",                                                      ""),
    (r"in\s+a\s+(?:real\s+)?sense\s*,?\s*",                                               ""),
    (r"in\s+(?:some|a)\s+way\s*,?\s*",                                                    ""),
    (r"so\s+to\s+speak\s*,?\s*",                                                           ""),
    (r"currently\s+right\s+now\s+at\s+this\s+(?:present\s+)?(?:moment|time)\s*,?\s*",    "Currently, "),
    (r"right\s+now\s+at\s+this\s+(?:present\s+)?(?:moment|time)\s*,?\s*",                "Currently, "),
    (r"at\s+this\s+(?:present\s+)?(?:moment|time)\s*,?\s*",                               "Currently, "),
    (r"right\s+now\s+currently\s*,?\s*",                                                   "Currently, "),
    (r"right\s+now\s*,?\s*",                                                               "now "),
]


def remove_redundant_hedges(text: str) -> tuple[str, list[dict]]:
    """
    Remove stacked hedges and redundant time/opinion markers.
    Returns (cleaned_text, list_of_changes).
    """
    result  = text
    changes = []

    for pattern, replacement in HEDGE_PATTERNS:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        for match in regex.finditer(result):
            original = match.group(0)
            repl     = replacement
            # Preserve sentence-start capitalisation
            if original.lstrip()[0:1].isupper() and repl:
                repl = repl[0].upper() + repl[1:]
            changes.append({
                "original":    original.strip(),
                "replacement": repl.strip() if repl.strip() else "(removed)",
            })
        result = regex.sub(
            lambda m: (replacement[0].upper() + replacement[1:]
                       if replacement and m.group(0).lstrip()[0:1].isupper()
                       else replacement),
            result
        )

    # Clean up
    result = re.sub(r" {2,}", " ", result)
    result = re.sub(r" ,",    ",", result)
    result = re.sub(r" \.",   ".", result)
    result = re.sub(r"\. ([a-z])", lambda m: ". " + m.group(1).upper(), result)
    return result.strip(), changes


# ---------------------------------------------------------------------------
# 4. Double negative detector & fixer
# ---------------------------------------------------------------------------

DOUBLE_NEGATIVES = [
    (r"\bdon't\s+(?:know|have|want|need|see|get|do)\s+nothing\b",  lambda m: m.group(0).replace("nothing","anything")),
    (r"\bdoesn't\s+(?:know|have|want|need|see|get|do)\s+nothing\b",lambda m: m.group(0).replace("nothing","anything")),
    (r"\bdidn't\s+\w+\s+nothing\b",                                lambda m: m.group(0).replace("nothing","anything")),
    (r"\bnobody\s+didn't\b",                                        lambda m: "nobody"),
    (r"\bcan't\s+\w+\s+nothing\b",                                  lambda m: m.group(0).replace("nothing","anything")),
    (r"\bnever\s+\w+\s+nobody\b",                                   lambda m: m.group(0).replace("nobody","anybody")),
    (r"\bnobody\s+don't\b",                                         lambda m: "nobody does"),
    (r"\bdon't\s+got\b",                                            lambda m: "don't have"),
    (r"\bain't\s+got\s+nothing\b",                                  lambda m: "have nothing"),
]


def fix_double_negatives(text: str) -> tuple[str, list[dict]]:
    """Fix double negatives in text."""
    result  = text
    changes = []
    for pattern, fixer in DOUBLE_NEGATIVES:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        for match in regex.finditer(result):
            original = match.group(0)
            fixed    = fixer(match)
            if fixed != original:
                changes.append({"original": original, "replacement": fixed})
        result = regex.sub(
            lambda m: fixer(m),
            result
        )
    return result, changes


# ---------------------------------------------------------------------------
# 5. Tautology checker (same idea twice in one sentence)
# ---------------------------------------------------------------------------

def detect_tautologies(text: str, threshold: float = 0.80) -> list[dict]:
    """
    Detect sentences that contain two clauses saying the same thing.
    e.g. "The world economy is important globally to the world"
    Splits each sentence on conjunctions and compares clause embeddings.
    """
    _ensure_models()
    sentences = nltk.sent_tokenize(text)
    found     = []

    split_pattern = re.compile(r'\b(?:and|but|because|since|while|although|which|that|,)\b', re.IGNORECASE)

    for sent in sentences:
        parts = [p.strip() for p in split_pattern.split(sent) if len(p.strip()) > 15]
        if len(parts) < 2:
            continue

        embeddings = _embed_model.encode(parts, show_progress_bar=False)
        for i in range(len(parts)):
            for j in range(i+1, len(parts)):
                sim = float(util.cos_sim(embeddings[i], embeddings[j]))
                if sim >= threshold:
                    found.append({
                        "sentence":    sent,
                        "clause_a":    parts[i],
                        "clause_b":    parts[j],
                        "similarity":  round(sim, 3),
                        "suggestion":  f'Remove or merge: "{parts[j]}" repeats the idea in "{parts[i]}"',
                        "type":        "tautology",
                    })

    return found


# ---------------------------------------------------------------------------
# Master function: run ALL smart fixes on a text
# ---------------------------------------------------------------------------

def smart_rewrite(
    text:          str,
    overused_words: list[dict] | None = None,
) -> dict:
    """
    Run all smart rewriting steps in order:
      1. Remove redundant hedges
      2. Fix double negatives
      3. Detect & remove pleonasms
      4. Replace overused words with synonyms
      5. Detect tautologies (report only, no auto-fix)

    Parameters
    ----------
    text           : corrected text (after GEC + redundancy fixes)
    overused_words : list from repetition.find_overused_words() — optional

    Returns
    -------
    dict:
        rewritten_text    : str — the improved text
        all_changes       : list[dict] — every change made
        pleonasms         : list — detected pleonasms
        tautologies       : list — detected tautologies (not auto-fixed)
        hedge_changes     : list
        neg_changes       : list
        synonym_changes   : list
        pleonasm_changes  : list
        summary           : str
    """
    _ensure_models()
    all_changes = []
    result      = text

    # Step 1: hedges
    result, hedge_ch = remove_redundant_hedges(result)
    all_changes.extend(hedge_ch)

    # Step 2: double negatives
    result, neg_ch = fix_double_negatives(result)
    all_changes.extend(neg_ch)

    # Step 3: pleonasms
    pleonasms        = detect_pleonasms(result)
    result, pleo_ch  = apply_pleonasm_fixes(result, pleonasms)
    all_changes.extend(pleo_ch)

    # Step 4: overused words
  # Step 4: overused words
# DISABLED TEMPORARILY — causes semantic corruption
    syn_ch = []

    # Step 5: tautologies (detect only)
    tautologies = detect_tautologies(result)

    # Build summary
    lines = []
    if hedge_ch:
        lines.append(f"  Redundant hedges removed ({len(hedge_ch)}):")
        for c in hedge_ch:
            lines.append(f"    ✗ '{c['original']}' → '{c['replacement']}'")
    if neg_ch:
        lines.append(f"  Double negatives fixed ({len(neg_ch)}):")
        for c in neg_ch:
            lines.append(f"    ✗ '{c['original']}' → '{c['replacement']}'")
    if pleo_ch:
        lines.append(f"  Pleonasms removed ({len(pleo_ch)}):")
        for c in pleo_ch:
            lines.append(f"    ✗ '{c['original']}' {c['replacement']}")
    if syn_ch:
        lines.append(f"  Overused words replaced ({len(syn_ch)}):")
        for c in syn_ch:
            lines.append(f"    ✗ '{c['original']}' → '{c['replacement']}' ({c['reason']})")
    if tautologies:
        lines.append(f"  Tautologies detected ({len(tautologies)}) — review manually:")
        for t in tautologies:
            lines.append(f"    ⚠ {t['suggestion']}")
    if not lines:
        lines.append("  No additional smart fixes applied.")

    return {
        "rewritten_text":  result,
        "all_changes":     all_changes,
        "pleonasms":       pleonasms,
        "tautologies":     tautologies,
        "hedge_changes":   hedge_ch,
        "neg_changes":     neg_ch,
        "synonym_changes": syn_ch,
        "pleonasm_changes":pleo_ch,
        "summary":         "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "In my personal opinion, I think that the global international world economy "
        "is a very major important issue because the world economy is important globally to the world. "
        "Currently right now at this present moment, me and my business partner goes to the financial bank "
        "because we needed to deposited some money funds for banking purposes. "
        "Their was a huge massive amount of many customers standing in the queue line, "
        "and the bank teller he was counting the cash money so slow that it make everyone "
        "waiting in the line very angry and mad. "
        "Everyone were complaining out loud with their voices about the slow banking delays, "
        "but nobody didn't do nothing about it to solve the problem."
    )

    from gec_project.repetition import find_overused_words
    overused = find_overused_words(sample)

    print("=== Smart Rewriter ===\n")
    result = smart_rewrite(sample, overused_words=overused)
    print("ORIGINAL:")
    print(sample)
    print("\nREWRITTEN:")
    print(result["rewritten_text"])
    print("\nCHANGES:")
    print(result["summary"])