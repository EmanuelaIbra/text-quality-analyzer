"""
Layer 2 + Layer 3 Redundancy Engine
-----------------------------------
Safe structural cleanup + clause compression
"""

import re
import spacy

_nlp = None


# ============================================================
# SPAcY
# ============================================================

def load_spacy():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")


def _ensure():
    if _nlp is None:
        load_spacy()


# ============================================================
# LAYER 2 — EXACT PATTERNS
# ============================================================

EXACT_PATTERNS = [
    (r'\breturn\s+back\b', 'return', 'scope collapse'),
    (r'\brepeat\s+again\b', 'repeat', 'scope collapse'),
    (r'\bfree\s+gift\b', 'gift', 'redundant noun'),
    (r'\bend\s+result\b', 'result', 'redundant noun'),
    (r'\bfinal\s+outcome\b', 'outcome', 'redundant noun'),
    (r'\bin\s+order\s+to\b', 'to', 'wordy phrase'),
]


def apply_exact_patterns(text: str):
    result = text
    changes = []

    for pattern, replacement, rule in EXACT_PATTERNS:
        regex = re.compile(pattern, flags=re.IGNORECASE)

        def repl(m):
            changes.append({
                "original": m.group(0),
                "replacement": replacement,
                "rule": rule
            })
            return replacement

        result = regex.sub(repl, result)

    return result, changes


# ============================================================
# PRONOUN FIX (LAYER 2)
# ============================================================

def find_pronoun_restatements(doc):
    issues = []

    for token in doc:
        if token.dep_ == "nsubj" and token.pos_ == "PRON":
            head = token.head
            for sib in head.lefts:
                if sib.dep_ in ("nsubj", "nsubjpass") and sib.pos_ in ("NOUN", "PROPN"):
                    issues.append((token, sib))

    return issues


def apply_pronoun_fixes(text: str):
    _ensure()
    doc = _nlp(text)

    issues = find_pronoun_restatements(doc)

    result = text
    changes = []

    for pronoun, noun in sorted(issues, key=lambda x: x[0].idx, reverse=True):

        pattern = re.compile(r'\s+\b' + re.escape(pronoun.text) + r'\b', re.IGNORECASE)

        new = pattern.sub("", result, count=1)

        if new != result:
            changes.append({
                "original": pronoun.text,
                "replacement": "(removed)",
                "rule": "pronoun restatement"
            })
            result = new

    return result, changes


# ============================================================
# LAYER 3 — CLAUSE EXTRACTION
# ============================================================

def extract_clauses(doc):
    clauses = []
    seen = set()

    for token in doc:
        if token.dep_ in ("ROOT", "conj", "relcl") and token.pos_ == "VERB":

            span_tokens = list(token.subtree)
            span = doc[span_tokens[0].i: span_tokens[-1].i + 1]

            key = (span.start, span.end)

            if key not in seen:
                seen.add(key)
                clauses.append(span.text)

    return clauses


# ============================================================
# REDUNDANCY SCORE (SAFE)
# ============================================================

def redundancy_score(clause: str):
    c = clause.lower()

    score = 0
    score += c.count(" and ")
    score += c.count(" that ")
    score += len(re.findall(r'\b(very|really|basically|actually)\b', c))
    score += len(re.findall(r'\b(\w+)\s+\1\b', c))
    score += len(clause.split()) // 18  # looser threshold

    return score


# ============================================================
# CLAUSE MERGING (SAFE)
# ============================================================

def get_subjects(doc):
    return [t.text.lower() for t in doc if t.dep_ == "nsubj"]


def merge_clauses(clauses):
    _ensure()

    docs = [_nlp(c) for c in clauses]

    used = set()
    merged = []

    for i, d1 in enumerate(docs):
        if i in used:
            continue

        base = clauses[i]
        subj1 = get_subjects(d1)

        for j, d2 in enumerate(docs):
            if i == j or j in used:
                continue

            subj2 = get_subjects(d2)

            if subj1 and subj1 == subj2:
                base = base.rstrip(".") + " and " + clauses[j].lower()
                used.add(j)

        merged.append(base)
        used.add(i)

    return merged


# ============================================================
# PRUNING (SAFE)
# ============================================================

def prune_clause(text):
    text = re.sub(r'\bthe fact that\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bin order to\b', 'to', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdue to the fact that\b', 'because', text, flags=re.IGNORECASE)
    return text.strip()


# ============================================================
# COORDINATION COMPRESSION
# ============================================================

def compress_coordination(text):
    patterns = [
        (r'\bfast\s+and\s+quick\b', 'fast'),
        (r'\bquick\s+and\s+fast\b', 'quick'),
        (r'\bhuge\s+and\s+massive\b', 'huge'),
        (r'\bmassive\s+and\s+huge\b', 'massive'),
    ]

    for p, r in patterns:
        text = re.sub(p, r, text, flags=re.IGNORECASE)

    return text


# ============================================================
# SAFE RECONSTRUCTION (CRITICAL FIX)
# ============================================================

def reconstruct(clauses):
    """
    FIX:
    - no naive join
    - prevents orphan fragments
    - prevents duplication
    """

    cleaned = []
    seen = set()

    for c in clauses:
        c = c.strip().rstrip(".")

        if len(c.split()) < 3:
            continue

        key = c.lower()

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(c)

    if not cleaned:
        return ""

    return ". ".join(cleaned) + "."


# ============================================================
# LAYER 2 PIPELINE
# ============================================================

def layer2_redundancy(text: str):
    text, c1 = apply_exact_patterns(text)
    text, c2 = apply_pronoun_fixes(text)

    return {
        "rewritten_text": text,
        "all_changes": c1 + c2,
    }


# ============================================================
# LAYER 3 PIPELINE (FIXED ORDER)
# ============================================================

def layer3_semantic_compressor(text: str):
    _ensure()

    doc = _nlp(text)

    clauses = extract_clauses(doc)

    # SAFE scoring filter
    scored = [(c, redundancy_score(c)) for c in clauses]

    kept = [c for c, s in scored if s <= 5] or clauses

    merged = merge_clauses(kept)

    pruned = [prune_clause(c) for c in merged]

    compressed = [compress_coordination(c) for c in pruned]

    final = reconstruct(compressed)

    return {
        "compressed_text": final,
        "clauses": clauses,
        "scored_clauses": scored,
    }