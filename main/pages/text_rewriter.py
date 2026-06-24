import re
import ollama


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation marks
    (., !, ?) followed by whitespace.

    Returns a cleaned list of non-empty sentences.
    """
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text.strip())
        if s.strip()
    ]


def _normalize(s: str) -> str:
    """
    Normalize whitespace by replacing multiple spaces
    with a single space and trimming leading/trailing spaces.

    Used throughout the module for reliable text comparison.
    """
    return re.sub(r"\s+", " ", s.strip())


class PreMerger:
    """
    Handles redundancy preparation before sending text
    to the LLM for rewriting.

    Responsibilities:
    - Identify highly similar sentence pairs.
    - Separate pairs requiring user decisions.
    - Identify merge candidates.
    - Track resolved repeated words.

    Similarity ranges:
    - >= user_choice_threshold:
        Requires user selection.
    - >= merge_threshold and < user_choice_threshold:
        Candidate for sentence merging.
    """

    def __init__(
            
        self,
        nlp=None,
        user_choice_threshold: float = 0.85,
        merge_threshold:       float = 0.65,
    ):
        """
        Initialize merger configuration.

        Args:
            nlp:
                Optional spaCy model.
            user_choice_threshold:
                Similarity score requiring explicit user choice.
            merge_threshold:
                Similarity score allowing automatic merge suggestions.
        """
        self.nlp                   = nlp
        self.user_choice_threshold = user_choice_threshold
        self.merge_threshold       = merge_threshold
        self.stop_words            = nlp.Defaults.stop_words if nlp else set()

    def merge(
        self,
        text:                str,
        redundant_sentences: list,
        repeated_words:      dict | list,
    ) -> tuple:
        """
        Analyze sentence redundancy and prepare rewrite metadata.

        Returns:
            (
                merged_text,
                resolved_repeats,
                merge_candidates,
                deleted_pairs,
                user_choice_candidates
            )

        Note:
        No sentences are automatically deleted.
        """

        # Normalize text to make matching consistent
        norm_text      = _normalize(text)

        # Split into normalized sentences
        sentences      = split_sentences(norm_text)

         # Fast lookup structure
        sentence_set   = {_normalize(s) for s in sentences}


        # Process highest similarity scores first
        pairs = sorted(redundant_sentences, key=lambda x: -x[2])

        merge_candidates       = []
        user_choice_candidates = []
        dropped                = set()

        for pair in pairs:
            sent_a = _normalize(pair[0])
            sent_b = _normalize(pair[1])
            score  = pair[2]

            # Skip if either sentence is no longer in the (normalised) text
            if sent_a not in sentence_set or sent_b not in sentence_set:
                continue
            # Highly similar sentences require user decision
            if score >= self.user_choice_threshold:
                user_choice_candidates.append({
                    "sentence_1": pair[0],
                    "sentence_2": pair[1],
                    "similarity": round(score, 2),
                    "action": (
                        "scelta_utente: scegliere la frase A, "
                        "la frase B oppure mantenere entrambe"
                    ),
                })

            elif self.merge_threshold <= score < self.user_choice_threshold:
                merge_candidates.append({
                    "sentence_1": pair[0],
                    "sentence_2": pair[1],
                    "similarity": round(score, 2),
                    "action": (
                        "unire solo se migliora la chiarezza; "
                        "non eliminare informazioni utili"
                    ),
                })
        # No automatic deletion currently occurs
        clean_sentences = [s for s in sentences if _normalize(s) not in dropped]

        # Collect repeated word lemmas into a flat set for resolved-repeat check
        if isinstance(repeated_words, dict):
            rw_set = set(repeated_words.keys())
        else:
            rw_set = {w for item in repeated_words for w in item.get("words", [])}

        resolved_repeats = self._find_resolved_repeats(
            dropped=dropped,
            remaining=clean_sentences,
            repeated_words=rw_set,
        )

        merged_text = " ".join(clean_sentences)

        return (
            merged_text,
            resolved_repeats,
            merge_candidates,
            [],                     # deleted_pairs — always empty (no auto-delete)
            user_choice_candidates,
        )

    def _content_words(self, sentence: str) -> set[str]:

        """
        Extract meaningful content words from a sentence.

        When spaCy is available:
            Uses lemmas and part-of-speech filtering.

        Fallback:
            Uses regex tokenization and stop-word removal.
        """
        if self.nlp:
            doc = self.nlp(sentence)
            return {
                t.lemma_.lower()
                for t in doc
                if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
                and not t.is_stop
                and not t.is_punct
            }
        tokens = re.findall(r"[a-zà-ÿ]+", sentence.lower())
        return {t for t in tokens if t not in self.stop_words}

    def _find_resolved_repeats(
        self,
        dropped:        set,
        remaining:      list[str],
        repeated_words: set[str],
    ) -> set[str]:
        """
        Identify repeated words that are no longer repeated
        after the merge preparation step.
        """
        remaining_text = " ".join(remaining).lower()
        return {
            word
            for word in repeated_words
            if len(re.findall(rf"\b{re.escape(word)}\b", remaining_text)) <= 1
        }


# ---------------------------------------------------------------------------
# LLM Output Cleanup
# ---------------------------------------------------------------------------

# Common introductory phrases generated by LLMs
# that should be removed before returning text.-------------------------------------------------------------------------

_PREAMBLE_PATTERNS = [
    r"^ecco(?: il)? testo riscritto[:\-]?\s*",
    r"^testo riscritto[:\-]?\s*",
    r"^versione corretta[:\-]?\s*",
    r"^versione migliorata[:\-]?\s*",
    r"^ho riscritto il testo[:\-]?\s*",
    r"^certo[,!]?\s*ecco[^:]*:\s*",
]


# Common explanatory sections that should be removed.
_POSTAMBLE_MARKERS = [
    r"^modifiche",
    r"^spiegazione",
    r"^nota",
    r"^ho rimosso",
    r"^ho unito",
    r"^ho corretto",
    r"^\*\s+",
    r"^-\s+",
]


def clean_llm_output(text: str) -> str:
    """
    Clean the raw response generated by the LLM.

    Removes:
    - Introductory phrases
    - Explanatory notes
    - Bullet lists
    - Excessive blank lines

    Returns only the rewritten text.
    """
    text = text.strip()
    for pattern in _PREAMBLE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    lines       = text.strip().splitlines()
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if any(re.match(p, stripped, re.IGNORECASE) for p in _POSTAMBLE_MARKERS):
            break
        clean_lines.append(line)

    text = "\n".join(clean_lines).strip()
    text = re.sub(r"^\s*[\*\-]\s+.+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Sei un revisore professionale di testi italiani. "
    "Riscrivi usando un italiano chiaro, naturale e semplice. "
    "Non usare parole troppo complesse, accademiche o artificiose se non sono già presenti nel testo originale. "
    "Conserva tutte le informazioni importanti. "
    "Non aggiungere nuovi fatti. "
    "Non eliminare informazioni quando non sei sicuro. "
    "Restituisci solo il testo riscritto."
    "Rimuovi parole e idee ripetute quando e neccessario"
)

_STYLE_MAP = {
    "concise":  "Rendi il testo più breve e chiaro, senza eliminare informazioni importanti.",
    "academic": "Usa uno stile formale e chiaro, evitando parole inutilmente complesse.",
    "fluent":   "Rendi il testo naturale e scorrevole, usando un lessico semplice.",
    "standard": "Usa un italiano chiaro, diretto e professionale.",
}


def build_prompt(
    pre_merged_text:        str,
    repetition_analysis:    dict,
    redundancy_report:      dict,
    resolved_repeats:       set,
    merge_candidates:       list,
    deleted_pairs:          list,
    user_choice_candidates: list,
    mode:                   str,
) -> str:
    """
    Build the final prompt sent to the LLM.

    This function combines:
    - Writing style instructions
    - Repetition analysis
    - Pleonasm detection results
    - Similar-word analysis
    - Redundant sentence analysis
    - User-choice candidates

    The goal is to provide the LLM with enough context to rewrite
    the text while preserving meaning and reducing redundancy.
    """
    style = _STYLE_MAP.get(mode, _STYLE_MAP["standard"])

    repeated_raw  = repetition_analysis.get("repeated_words", {})
    still_repeated = {
        w: c for w, c in (
            repeated_raw.items() if isinstance(repeated_raw, dict)
            else {w: 2 for item in repeated_raw for w in item.get("words", [])}.items()
        )
        if c >= 2 and w not in resolved_repeats
    }
    # Convert repeated words into a readable string for the prompt.
    repeated_str = ", ".join(
        f"'{w}' (x{c})" for w, c in still_repeated.items()
    ) or "nessuna"


  # Format detected pleonasms and their suggested replacements.
    pleonasm_str = "\n".join(
        f"  '{p['phrase']}' → '{p['replacement']}'"
        for p in redundancy_report.get("pleonasms", [])
    ) or "  nessuno"

# ------------------------------------------------------------------
    # Similar words
    # ------------------------------------------------------------------
    # Include highly similar words (potential synonyms or redundancies)
    # that the LLM may want to simplify or reduce.
    #
    # Prompt length is capped to avoid excessive context.
    sim_pairs_str = "\n".join(
        f"  '{a}' e '{b}' (punteggio {s:.2f})"
        for a, b, s in redundancy_report.get("similar_words", [])
        if 0.75 <= s < 1.00
    )[:6 * 60] or "  nessuna"       # rough cap on prompt length
     # ------------------------------------------------------------------
    # Deleted sentence pairs
    # ------------------------------------------------------------------
    # Sentences already removed before rewriting.
    # Useful as context so the LLM doesn't accidentally recreate them.
    deleted_str = "  nessuna"
    if deleted_pairs:
        deleted_str = ""
        for idx, item in enumerate(deleted_pairs, 1):
            deleted_str += (
                f"\n  Coppia {idx} (similarità {item['similarity']}):\n"
                f"    Mantenuta: {item['kept']}\n"
                f"    Rimossa: {item['removed']}\n"
            )
    # ------------------------------------------------------------------
    # Merge candidates
    # ------------------------------------------------------------------
    # Sentence pairs that are similar enough to be merged but not
    # similar enough to require an explicit user choice.
    merge_str = "  nessuna"
    if merge_candidates:
        merge_str = ""
        for idx, item in enumerate(merge_candidates, 1):
            merge_str += (
                f"\n  Coppia {idx} (similarità {item['similarity']}):\n"
                f"    A: {item['sentence_1']}\n"
                f"    B: {item['sentence_2']}\n"
                f"    Azione: {item['action']}\n"
            )
# ------------------------------------------------------------------
    # User-choice candidates
    # ------------------------------------------------------------------
    # Highly similar sentence pairs that require explicit user approval
    # before one can be removed.
    user_choice_str = "  nessuna"
    if user_choice_candidates:
        user_choice_str = ""
        for idx, item in enumerate(user_choice_candidates, 1):
            user_choice_str += (
                f"\n  Coppia {idx} (similarità {item['similarity']}):\n"
                f"    A: {item['sentence_1']}\n"
                f"    B: {item['sentence_2']}\n"
                "    Azione: non eliminare automaticamente; "
                "se non esiste una scelta esplicita dell'utente, mantieni entrambe "
                "oppure fondile senza perdere informazioni.\n"
            )

    return f"""Stile di riscrittura:
{style}

ANALISI USATA PRIMA DELLA RISCRITTURA:

1. Parole ancora ripetute:
   {repeated_str}

2. Pleonasmi da rimuovere:
{pleonasm_str}

3. Parole simili o quasi sinonimi:
{sim_pairs_str}

4. Frasi duplicate già rimosse automaticamente:
{deleted_str}

5. Frasi correlate che possono essere unite:
{merge_str}

6. Frasi molto simili che richiedono scelta dell'utente:
{user_choice_str}

REGOLE:
- Usa l'analisi sopra per guidare la riscrittura.
- Non eliminare automaticamente le frasi indicate come scelta dell'utente.
- Se l'utente non ha scelto, mantieni entrambe oppure fondile senza perdere informazioni.
- Se sono presenti frasi correlate, uniscile solo quando migliora la chiarezza.
- Rimuovi parole e idee ripetute.
- Rimuovi i pleonasmi.
- Usa un italiano semplice e naturale.
- Non rendere il testo troppo elegante o artificioso.
- Non usare sinonimi difficili solo per variare.
- Mantieni tutte le informazioni importanti.
- Non aggiungere nuove informazioni.
- Restituisci solo il testo finale riscritto.
- Non modificare, eliminare o spostare i placeholder nel formato [[KEEP_SENTENCE_X]].

TESTO:
{pre_merged_text.strip()}"""


# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------

class TextRewriter:
    """
    Responsible for generating the final rewritten version of a text.

    Workflow:
    1. Pre-process the text using PreMerger.
    2. Identify:
       - resolved repetitions
       - merge candidates
       - user-choice sentence pairs
       - deleted sentence pairs
    3. Build a detailed prompt containing all analysis results.
    4. Send the prompt to the LLM (Ollama).
    5. Clean the LLM response.
    6. Return the final rewritten text.
    """
    def __init__(
        self,
        model:                 str   = "llama3.1",
        nlp                         = None,
        user_choice_threshold: float = 0.85,
        merge_threshold:       float = 0.65,
    ):
        """
        Initialize the text rewriter.

        Args:
            model:
                Ollama model used for rewriting.

            nlp:
                Optional spaCy model passed to PreMerger.

            user_choice_threshold:
                Similarity score above which sentence pairs
                require explicit user selection.

            merge_threshold:
                Similarity score above which sentence pairs
                become merge candidates.
        """
        self.model  = model
        self.merger = PreMerger(
            nlp=nlp,
            user_choice_threshold=user_choice_threshold,
            merge_threshold=merge_threshold,
        )

    def rewrite(
        self,
        text:                str,
        repetition_analysis: dict,
        redundancy_report:   dict,
        mode:                str = "concise",
    ) -> str:
        """
        Rewrite text using redundancy and repetition analysis.

        Args:
            text:
                Original text to rewrite.

            repetition_analysis:
                Results from the repetition analyzer.

            redundancy_report:
                Results from redundancy, pleonasm,
                and similarity detection.

            mode:
                Rewriting style:
                - concise
                - fluent
                - academic
                - standard

        Returns:
            Cleaned rewritten text generated by the LLM.
        """

        # -------------------------------------------------------------
        # Pre-processing phase
        # -------------------------------------------------------------
        # Analyze sentence redundancy before rewriting.
        #
        # Returns:
        #   pre_merged              -> text after preprocessing
        #   resolved_repeats        -> repetitions already resolved
        #   merge_candidates        -> sentence pairs that may be merged
        #   deleted_pairs           -> removed duplicate pairs
        #   user_choice_candidates  -> pairs requiring user decisions
        (
            pre_merged,
            resolved_repeats,
            merge_candidates,
            deleted_pairs,
            user_choice_candidates,
        ) = self.merger.merge(
            text=text,
            redundant_sentences=redundancy_report.get("redundant_sentences", []),
            repeated_words=repetition_analysis.get("repeated_words", {}),
        )
         # -------------------------------------------------------------
        # Prompt construction
        # -------------------------------------------------------------
        # Build a structured prompt containing:
        #   - style instructions
        #   - repetition analysis
        #   - pleonasm analysis
        #   - redundancy analysis
        #   - merge suggestions
        #   - user-choice constraints
        #
        # This provides the LLM with all contextual information
        # needed to rewrite the text safely.
        prompt = build_prompt(
            pre_merged_text=pre_merged,
            repetition_analysis=repetition_analysis,
            redundancy_report=redundancy_report,
            resolved_repeats=resolved_repeats,
            merge_candidates=merge_candidates,
            deleted_pairs=deleted_pairs,
            user_choice_candidates=user_choice_candidates,
            mode=mode,
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0.1},
        )

        return clean_llm_output(response["message"]["content"])