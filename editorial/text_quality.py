"""
editorial/text_quality.py
--------------------------
Text quality indexes:

  1. Lexical Diversity Index (LDIX)
       LDIX = unique_words / total_words
       Range [0, 1] — higher = richer vocabulary

  2. Syntactic Complexity Index (SCIX)
       SCIX = MLS × (S/C Ratio) × SR
       where:
         MLS         = Mean Length of Sentence (tokens)
         S/C Ratio   = Sentences / Clauses  (higher = simpler structure)
         SR          = Subordination Ratio  = subordinate clauses / total clauses

  3. Readability — Flesch Reading Ease
       FRE = 206.835 - 1.015×(words/sentences) - 84.6×(syllables/words)
       Range [0, 100] — higher = easier to read
       (English equivalent of Gulpease; works for any language)

  4. Syntactic Balance Score
       Checks sentence length variance — high variance = unbalanced writing.
       Also detects: passive voice ratio, avg subordinate clauses per sentence.

  5. Base Vocabulary Check
       Flags words outside the most common 5000 English words (Oxford list).
       Helps identify vocabulary that may be too advanced or too rare.

All scores are computed on both ORIGINAL and CORRECTED text so you can
see the improvement after GEC + redundancy fixes.
"""

import re
import math
import collections
import spacy
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/cmudict")
except LookupError:
    nltk.download("cmudict", quiet=True)


# ---------------------------------------------------------------------------
# spaCy loader
# ---------------------------------------------------------------------------

_nlp = None

def _ensure_spacy():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError("Run: python -m spacy download en_core_web_sm")


# ---------------------------------------------------------------------------
# Syllable counter (for Flesch)
# ---------------------------------------------------------------------------

try:
    from nltk.corpus import cmudict as _cmudict
    _CMUDICT = _cmudict.dict()
except Exception:
    _CMUDICT = {}


def _count_syllables(word: str) -> int:
    """Count syllables using CMU dict, fall back to vowel-group heuristic."""
    w = word.lower().strip(".,!?;:'\"")
    if w in _CMUDICT:
        # count vowel phonemes
        return sum(1 for ph in _CMUDICT[w][0] if ph[-1].isdigit())
    # heuristic fallback
    vowels = re.findall(r'[aeiouy]+', w)
    count  = len(vowels)
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


# ---------------------------------------------------------------------------
# 1. Lexical Diversity Index (LDIX)
# ---------------------------------------------------------------------------

def lexical_diversity(text: str) -> dict:
    """
    LDIX = unique_lemmas / total_content_tokens

    We use lemmas (so 'running', 'ran', 'runs' count as one type)
    and restrict to content words (nouns, verbs, adjectives, adverbs)
    for a more meaningful score.

    Returns
    -------
    dict:
        ldix            : float [0,1]
        total_tokens    : int
        unique_lemmas   : int
        top_words       : list[(lemma, count)] top 10 most used
        label           : str  'high' / 'medium' / 'low'
        interpretation  : str
    """
    _ensure_spacy()
    doc = _nlp(text)

    content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    tokens = [
        t.lemma_.lower() for t in doc
        if t.pos_ in content_pos
        and not t.is_stop
        and not t.is_punct
        and len(t.lemma_) > 2
    ]

    if not tokens:
        return {"ldix": 0.0, "total_tokens": 0, "unique_lemmas": 0,
                "top_words": [], "label": "N/A", "interpretation": "No content words found."}

    total   = len(tokens)
    unique  = len(set(tokens))
    ldix    = round(unique / total, 4)
    freq    = collections.Counter(tokens)

    if ldix >= 0.75:
        label = "high"
        interp = "Excellent vocabulary variety. Text uses diverse, non-repetitive language."
    elif ldix >= 0.50:
        label = "medium"
        interp = "Moderate vocabulary variety. Some words are repeated — consider synonyms."
    else:
        label = "low"
        interp = "Low vocabulary variety. Many words are overused — introduce more synonyms."

    return {
        "ldix":          ldix,
        "total_tokens":  total,
        "unique_lemmas": unique,
        "top_words":     freq.most_common(10),
        "label":         label,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# 2. Syntactic Complexity Index (SCIX)
# ---------------------------------------------------------------------------

# Clause-introducing dependency labels in spaCy
CLAUSE_DEPS = {"relcl", "advcl", "ccomp", "xcomp", "acl"}
SUBORD_DEPS = {"relcl", "advcl", "ccomp", "acl"}   # subordinate subset


def syntactic_complexity(text: str) -> dict:
    """
    SCIX = MLS × (S/C Ratio) × SR

    MLS       = mean sentence length in tokens
    S/C Ratio = num_sentences / num_clauses  (lower = more complex)
    SR        = subordinate_clauses / total_clauses

    Also reports:
      - passive voice ratio
      - avg clauses per sentence
      - sentence length std deviation (balance indicator)

    Returns
    -------
    dict with scix, mls, sc_ratio, sr, passive_ratio, label, interpretation,
         sentence_lengths, length_std, suggestions
    """
    _ensure_spacy()
    doc = _nlp(text)

    sentences     = list(doc.sents)
    num_sentences = len(sentences)
    if num_sentences == 0:
        return {"scix": 0.0, "label": "N/A", "interpretation": "No sentences found."}

    sent_lengths   = [len([t for t in s if not t.is_punct and not t.is_space])
                      for s in sentences]
    mls            = sum(sent_lengths) / num_sentences

    # Count clauses and passive constructions
    total_clauses  = 0
    subord_clauses = 0
    passive_count  = 0

    for sent in sentences:
        # Root counts as 1 clause
        total_clauses += 1
        for token in sent:
            if token.dep_ in CLAUSE_DEPS:
                total_clauses += 1
            if token.dep_ in SUBORD_DEPS:
                subord_clauses += 1
            # Passive: auxpass dependency or "by" following a past participle
            if token.dep_ == "auxpass":
                passive_count += 1

    sc_ratio = num_sentences / total_clauses if total_clauses > 0 else 1.0
    sr       = subord_clauses / total_clauses if total_clauses > 0 else 0.0
    scix     = round(mls * sc_ratio * sr, 4)

    passive_ratio = round(passive_count / num_sentences, 4)

    # Sentence length standard deviation (balance)
    mean_len  = sum(sent_lengths) / len(sent_lengths)
    variance  = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
    length_std = round(math.sqrt(variance), 2)

    # Label and interpretation
    if scix < 1.0:
        label  = "simple"
        interp = "Simple sentence structure. Consider adding more variety with complex/compound sentences."
    elif scix < 3.0:
        label  = "moderate"
        interp = "Moderate syntactic complexity. Good balance between simple and complex structures."
    else:
        label  = "complex"
        interp = "High syntactic complexity. Consider simplifying some sentences for readability."

    # Balance label
    if length_std < 5:
        balance = "well-balanced"
    elif length_std < 10:
        balance = "slightly uneven"
    else:
        balance = "unbalanced — high variance in sentence length"

    # Suggestions
    suggestions = []
    if passive_ratio > 0.3:
        suggestions.append(
            f"High passive voice usage ({passive_count} sentences). "
            "Convert to active voice for clearer writing."
        )
    if length_std > 10:
        suggestions.append(
            f"Sentence lengths vary widely (std={length_std}). "
            "Mix short and long sentences more evenly."
        )
    if mls > 30:
        suggestions.append(
            f"Average sentence length is {mls:.1f} words — quite long. "
            "Consider splitting some sentences."
        )
    if mls < 8:
        suggestions.append(
            f"Average sentence length is {mls:.1f} words — very short. "
            "Consider combining short sentences for better flow."
        )

    return {
        "scix":             scix,
        "mls":              round(mls, 2),
        "sc_ratio":         round(sc_ratio, 4),
        "sr":               round(sr, 4),
        "total_clauses":    total_clauses,
        "subord_clauses":   subord_clauses,
        "passive_ratio":    passive_ratio,
        "passive_count":    passive_count,
        "sentence_lengths": sent_lengths,
        "length_std":       length_std,
        "label":            label,
        "balance":          balance,
        "interpretation":   interp,
        "suggestions":      suggestions,
    }


# ---------------------------------------------------------------------------
# 3. Flesch Reading Ease (Readability)
# ---------------------------------------------------------------------------

def flesch_reading_ease(text: str) -> dict:
    """
    FRE = 206.835 - 1.015×(words/sentences) - 84.6×(syllables/words)

    Score interpretation:
      90-100 : Very easy  (5th grade)
      70-90  : Easy
      60-70  : Standard
      50-60  : Fairly difficult
      30-50  : Difficult
      0-30   : Very difficult (college graduate level)

    Returns dict with score, label, interpretation, avg_sentence_length,
    avg_syllables_per_word.
    """
    sentences = nltk.sent_tokenize(text)
    words     = nltk.word_tokenize(text)
    words     = [w for w in words if w.isalpha()]

    num_sentences = len(sentences)
    num_words     = len(words)
    num_syllables = sum(_count_syllables(w) for w in words)

    if num_sentences == 0 or num_words == 0:
        return {"fre": 0.0, "label": "N/A", "interpretation": "Not enough text."}

    asl  = num_words / num_sentences          # avg sentence length
    asw  = num_syllables / num_words          # avg syllables per word
    fre  = 206.835 - 1.015 * asl - 84.6 * asw
    fre  = round(max(0.0, min(100.0, fre)), 2)

    if fre >= 90:
        label  = "Very easy"
        interp = "Reads like a children's book. Fine for casual writing; may be too simple for academic work."
    elif fre >= 70:
        label  = "Easy"
        interp = "Easy to read. Suitable for most audiences."
    elif fre >= 60:
        label  = "Standard"
        interp = "Plain English. Good for general audiences."
    elif fre >= 50:
        label  = "Fairly difficult"
        interp = "Requires some effort to read. Suitable for educated general audiences."
    elif fre >= 30:
        label  = "Difficult"
        interp = "Hard to read. Suitable for academic or professional writing."
    else:
        label  = "Very difficult"
        interp = "Very complex. Suitable only for specialist or academic audiences."

    return {
        "fre":                    fre,
        "label":                  label,
        "interpretation":         interp,
        "avg_sentence_length":    round(asl, 2),
        "avg_syllables_per_word": round(asw, 2),
        "num_sentences":          num_sentences,
        "num_words":              num_words,
        "num_syllables":          num_syllables,
    }


# ---------------------------------------------------------------------------
# 4. Base Vocabulary Check
# ---------------------------------------------------------------------------

# Top ~2000 most common English words (Oxford/Longman simplified list)
# This is a curated subset — enough for useful flagging without false positives
BASE_VOCAB = {
    "the","be","to","of","and","a","in","that","have","it","for","not","on",
    "with","he","as","you","do","at","this","but","his","by","from","they",
    "we","say","her","she","or","an","will","my","one","all","would","there",
    "their","what","so","up","out","if","about","who","get","which","go","me",
    "when","make","can","like","time","no","just","him","know","take","people",
    "into","year","your","good","some","could","them","see","other","than",
    "then","now","look","only","come","its","over","think","also","back","after",
    "use","two","how","our","work","first","well","way","even","new","want",
    "because","any","these","give","day","most","us","great","between","need",
    "large","often","hand","high","place","hold","turn","help","too","such",
    "feel","keep","children","begin","got","walk","example","ease","paper",
    "group","always","music","those","both","mark","book","letter","until",
    "mile","river","car","feet","care","second","enough","plain","girl","usual",
    "young","ready","above","ever","red","list","though","feel","talk","bird",
    "soon","body","dog","family","direct","pose","leave","song","measure","door",
    "product","black","short","numeral","class","wind","question","happen",
    "complete","ship","area","half","rock","order","fire","south","problem",
    "piece","told","knew","pass","since","top","whole","king","space","heard",
    "best","hour","better","true","during","hundred","five","remember","step",
    "early","hold","west","ground","interest","reach","fast","verb","sing",
    "listen","six","table","travel","less","morning","ten","simple","several",
    "vowel","toward","war","lay","against","pattern","slow","center","love",
    "person","money","serve","appear","road","map","rain","rule","govern",
    "pull","cold","notice","voice","unit","power","town","fine","drive","ran",
    "don","determine","study","break","nature","realize","catch","clear","mouth",
    "size","open","seem","together","next","white","children","begin","got",
    "ago","stood","plane","system","behind","run","round","boat","game","force",
    "bring","understand","warm","common","bring","explain","dry","though","language",
    "shape","deep","thousands","yes","clear","equation","yet","government","filled",
    "heat","full","hot","check","object","am","rule","among","noun","power",
    "cannot","able","six","dark","ball","material","special","heavy","fine",
    "pair","circle","include","built","can't","won't","don't","it's","that's",
    # Academic / connector words kept in base
    "however","therefore","although","because","since","while","whether",
    "another","different","possible","important","following","provide",
    "through","process","result","increase","specific","suggest","require",
    "consider","various","general","particular","according","example","research",
    "information","analysis","method","approach","significant","develop",
    "identify","establish","evidence","determine","indicate","represent",
    "discuss","describe","explain","compare","show","argue","conclude",
}


def base_vocabulary_check(text: str, flag_threshold: int = 3) -> dict:
    """
    Flag words that appear >= flag_threshold times AND are NOT in the base vocabulary.
    These are either advanced/technical words or potentially misspelled rare words.

    Returns
    -------
    dict:
        advanced_words  : list of (word, count) — used repeatedly, not in base vocab
        rare_words      : list of words used only once, not in base vocab
        base_coverage   : float — % of tokens that are in base vocab
        total_unique    : int
    """
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    if not words:
        return {"advanced_words": [], "rare_words": [], "base_coverage": 0.0, "total_unique": 0}

    freq  = collections.Counter(words)
    total = len(words)

    in_base    = sum(1 for w in words if w in BASE_VOCAB)
    coverage   = round(in_base / total, 4)

    advanced   = [(w, c) for w, c in freq.most_common()
                  if w not in BASE_VOCAB and c >= flag_threshold]
    rare       = [w for w, c in freq.items()
                  if w not in BASE_VOCAB and c == 1 and len(w) > 6][:15]

    return {
        "advanced_words": advanced[:15],
        "rare_words":     rare,
        "base_coverage":  coverage,
        "total_unique":   len(freq),
    }


# ---------------------------------------------------------------------------
# 5. Full text quality analysis  (runs all four)
# ---------------------------------------------------------------------------

def full_text_quality(text: str, label: str = "") -> dict:
    """
    Run all text quality indexes on a text.

    Parameters
    ----------
    text  : the text to analyse
    label : optional label like "original" or "corrected" for reporting

    Returns
    -------
    dict with ldix, scix, fre, base_vocab, summary string
    """
    ld   = lexical_diversity(text)
    sc   = syntactic_complexity(text)
    fre  = flesch_reading_ease(text)
    bv   = base_vocabulary_check(text)

    tag  = f" [{label}]" if label else ""

    lines = [f"TEXT QUALITY INDEXES{tag}", ""]

    # LDIX
    lines.append(f"  Lexical Diversity (LDIX) : {ld['ldix']:.4f}  [{ld['label']}]")
    lines.append(f"    {ld['interpretation']}")
    if ld["top_words"]:
        top = ", ".join(f"'{w}'×{c}" for w, c in ld["top_words"][:5])
        lines.append(f"    Most used: {top}")

    lines.append("")

    # SCIX
    lines.append(f"  Syntactic Complexity (SCIX): {sc['scix']:.4f}  [{sc['label']}]")
    lines.append(f"    MLS={sc['mls']} words/sentence  |  Clauses/sent={sc.get('total_clauses',0)/max(1,len(sc.get('sentence_lengths',[1]))):.1f}")
    lines.append(f"    Passive voice: {sc['passive_count']} sentence(s)  |  Balance: {sc['balance']}")
    lines.append(f"    {sc['interpretation']}")
    for s in sc["suggestions"]:
        lines.append(f"    ⚠ {s}")

    lines.append("")

    # FRE
    lines.append(f"  Readability (Flesch)      : {fre['fre']:.1f} / 100  [{fre['label']}]")
    lines.append(f"    Avg sentence length: {fre['avg_sentence_length']} words")
    lines.append(f"    Avg syllables/word : {fre['avg_syllables_per_word']}")
    lines.append(f"    {fre['interpretation']}")

    lines.append("")

    # Base vocabulary
    lines.append(f"  Base Vocab Coverage       : {bv['base_coverage']*100:.1f}%")
    if bv["advanced_words"]:
        adv = ", ".join(f"'{w}'×{c}" for w, c in bv["advanced_words"][:6])
        lines.append(f"    Frequently used advanced words: {adv}")
    if bv["rare_words"]:
        lines.append(f"    Rare/complex words: {', '.join(bv['rare_words'][:8])}")

    return {
        "ldix":       ld,
        "scix":       sc,
        "fre":        fre,
        "base_vocab": bv,
        "summary":    "\n".join(lines),
    }


def compare_quality(original: str, corrected: str) -> dict:
    """
    Run full_text_quality on both texts and return a comparison dict
    with delta values for each index.
    """
    orig_q = full_text_quality(original,  label="original")
    corr_q = full_text_quality(corrected, label="corrected")

    delta_ldix = round(corr_q["ldix"]["ldix"] - orig_q["ldix"]["ldix"], 4)
    delta_fre  = round(corr_q["fre"]["fre"]   - orig_q["fre"]["fre"],   2)
    delta_scix = round(corr_q["scix"]["scix"] - orig_q["scix"]["scix"], 4)

    return {
        "original":   orig_q,
        "corrected":  corr_q,
        "delta_ldix": delta_ldix,
        "delta_fre":  delta_fre,
        "delta_scix": delta_scix,
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "In my personal opinion, I think that a basic fundamental understanding "
        "of math is absolutely essential and completely necessary for all students. "
        "Each and every day on a daily basis, we encounter and come across various "
        "different math problems in our normal everyday lives. "
        "For this reason, it is vitally important that schools must continue to "
        "always teach these skills to children who are young. "
        "If we look back in retrospect at the past history of education, math has "
        "always been an indispensable requirement that is mandatory, and it will "
        "continue to remain that way ahead in the future."
    )

    result = full_text_quality(sample, label="original")
    print(result["summary"])