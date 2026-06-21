import re
import spacy
import ollama


try:
    _NLP = spacy.load("it_core_news_lg")
except OSError:
    _NLP = None


def split_sentences(text):
    """
    Splits text into clean individual sentences.
    Prioritizes spaCy's statistical sentence segmenter; falls back to a 
    regex-based split on punctuation boundaries if spaCy is missing.
    """
    if _NLP:
        doc = _NLP(text)
        return [
            sentence.text.strip()
            for sentence in doc.sents
            if sentence.text.strip()
        ]

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


class PreMerger:
    """
    Pre-processes text to evaluate redundant sentences prior to LLM submission.
    Deduplicates high-similarity pairs and flags moderate-similarity pairs for conditional merging.
    """
    def __init__(
        self,
        delete_threshold=0.96,
        merge_threshold=0.88,
    ):
        self.delete_threshold = delete_threshold
        self.merge_threshold = merge_threshold

        if _NLP:
            self.stop_words = _NLP.Defaults.stop_words
        else:
            self.stop_words = set()

    def merge(self, text, redundant_sentences, repeated_words):
        """
        Main filtering routine. Drops sentences that surpass the deletion threshold 
        and groups sentences that meet the merge criteria.
        """
        sentences = split_sentences(text)
        # Sort incoming redundant pairs descending by their similarity score
        pairs = sorted(
            redundant_sentences,
            key=lambda item: -item[2],
        )

        dropped = set()
        deleted_pairs = []
        merge_candidates = []

        for sent_a, sent_b, score in pairs:
            # Skip if either sentence in the pair has already been marked for deletion
            if sent_a in dropped or sent_b in dropped:
                continue

            # Ensure both sentences still physically exist in our working sentence array
            if sent_a not in sentences or sent_b not in sentences:
                continue

            # Case 1: Extreme similarity -> Drop one of the sentences automatically
            if score >= self.delete_threshold:
                to_drop = self._choose_drop(sent_a, sent_b)
                to_keep = sent_b if to_drop == sent_a else sent_a

                dropped.add(to_drop)

                deleted_pairs.append({
                    "kept": to_keep,
                    "removed": to_drop,
                    "similarity": round(score, 2),
                })
            # Case 2: Moderate similarity -> Flag as a candidate for the LLM to conditionally merge
            elif self.merge_threshold <= score < self.delete_threshold:
                merge_candidates.append({
                    "sentence_1": sent_a,
                    "sentence_2": sent_b,
                    "similarity": round(score, 2),
                    "action": (
                        "unire solo se migliora la chiarezza; "
                        "non eliminare informazioni utili"
                    ),
                })
        # Rebuild the sentence list, filtering out all dropped duplicates
        clean_sentences = [
            sentence
            for sentence in sentences
            if sentence not in dropped
        ]
        # Determine which word repetitions were naturally resolved by dropping sentences
        resolved_repeats = self._find_resolved_repeats(
            dropped=dropped,
            remaining=clean_sentences,
            repeated_words=repeated_words,
        )

        merged_text = " ".join(clean_sentences)

        return (
            merged_text,
            resolved_repeats,
            merge_candidates,
            deleted_pairs,
        )

    def _content_words(self, sentence):
        """
        Isolates and returns the set of lowercase base lemmas (or words) from a sentence,
        excluding structural elements, stop words, and punctuation.
        """
        if _NLP:
            doc = _NLP(sentence)

            return {
                token.lemma_.lower()
                for token in doc
                if (
                    token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
                    and not token.is_stop
                    and not token.is_punct
                )
            }

        tokens = re.findall(r"[a-zà-ÿ]+", sentence.lower())

        return {
            token
            for token in tokens
            if token not in self.stop_words
        }

    def _choose_drop(self, sent_a, sent_b):
        """
        Heuristic to decide which sentence to drop. It eliminates subsets, 
        or defaults to dropping the shorter sentence to prioritize information preservation.
        """
        words_a = self._content_words(sent_a)
        words_b = self._content_words(sent_b)   

        # If sentence A's content is entirely contained within sentence B, drop A
        if words_a <= words_b:
            return sent_a
        
        # If sentence B's content is entirely contained within sentence A, drop B
        if words_b <= words_a:
            return sent_b
        
        # Default fallback: Drop whichever sentence has fewer content words
        return sent_a if len(words_a) <= len(words_b) else sent_b

    def _find_resolved_repeats(self, dropped, remaining, repeated_words):
        """
        Recalculates word frequencies across the remaining clean text to see 
        which global word repetitions were resolved by the sentence drops.
        """
        remaining_text = " ".join(remaining).lower()
        resolved = set()

        for word in repeated_words:
            # Count remaining explicit boundary matches of the target word
            count = len(
                re.findall(
                    rf"\b{re.escape(word)}\b",
                    remaining_text,
                )
            )
            # If the word now appears at most once, its redundancy is resolved
            if count <= 1:
                resolved.add(word)

        return resolved

# Pre-compiled regular expressions to strip out intro phrases frequently generated by LLMs
_PREAMBLE_PATTERNS = [
    r"^ecco(?: il)? testo riscritto[:\-]?\s*",
    r"^testo riscritto[:\-]?\s*",
    r"^versione corretta[:\-]?\s*",
    r"^versione migliorata[:\-]?\s*",
    r"^ho riscritto il testo[:\-]?\s*",
    r"^certo[,!]?\s*ecco[^:]*:\s*",
]

# Regular expressions to spot where structural explanation notes or change logs begin in LLM outputs
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


def clean_llm_output(text):
    """
    Post-processes the raw text returned by the language model. Strips conversational 
    preambles, explanations, lists, and extra spacing to ensure a clean result.
    """
    text = text.strip()

# Iteratively remove conversational introductions matching our preamble list
    for pattern in _PREAMBLE_PATTERNS:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        )

    lines = text.strip().splitlines()
    clean_lines = []

   # Iterate over lines and immediately discard everything if an explanation marker is reached
    for line in lines:
        stripped = line.strip()

        if any(
            re.match(pattern, stripped, re.IGNORECASE)
            for pattern in _POSTAMBLE_MARKERS
        ):
            break

        clean_lines.append(line)

    text = "\n".join(clean_lines).strip()
    # Wipe out any bulleted item configurations that survived the sweep
    text = re.sub(r"^\s*[\*\-]\s+.+$", "", text, flags=re.MULTILINE)
    # Collapse multiple consecutive empty vertical newlines down to a max of two
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# Definition of the base persona rules assigned to the model via the system role
SYSTEM_PROMPT = (
    "Sei un revisore professionale di testi italiani. "
    "Riscrivi usando un italiano chiaro, naturale e semplice. "
    "Non usare parole troppo complesse, accademiche o artificiose se non sono già presenti nel testo originale. "
    "Conserva le informazioni importanti. "
    "Non aggiungere nuovi fatti. "
    "Restituisci solo il testo riscritto."
)


def build_prompt(
    pre_merged_text,
    repetition_analysis,
    redundancy_report,
    resolved_repeats,
    merge_candidates,
    deleted_pairs,
    mode,
):
    """
    Dynamically constructs a highly structural, information-dense instruction prompt.
    It passes the structural analysis data and styling preferences into the LLM context.
    """
    # Select the precise style modifier instruction string according to selected mode
    style = {
        "concise": (
            "Rendi il testo più breve e chiaro, senza eliminare informazioni importanti."
        ),
        "academic": (
            "Usa uno stile formale e chiaro, evitando parole inutilmente complesse."
        ),
        "fluent": (
            "Rendi il testo naturale e scorrevole, usando un lessico semplice."
        ),
        "standard": (
            "Usa un italiano chiaro, diretto e professionale."
        ),
    }.get(mode, "Usa un italiano chiaro, diretto e professionale.")

    # Identify words that are still duplicated after pre-merging adjustments
    repeated_raw = repetition_analysis.get("repeated_words", {})

    still_repeated = {
        word: count
        for word, count in repeated_raw.items()
        if count >= 2 and word not in resolved_repeats
    }

    repeated_str = ", ".join(
        f"'{word}' (x{count})"
        for word, count in still_repeated.items()
    ) or "nessuna"

    pleonasm_items = redundancy_report.get("pleonasms", [])

    pleonasm_str = "\n".join(
       f"  '{item['phrase']}' → '{item['replacement']}'"
       for item in pleonasm_items
    ) or "  nessuno"

    raw_pairs = redundancy_report.get("similar_words", [])

   # Filter out semantic similarity sets, capturing the top 6 pairs between 0.75 and 1.00 similarity
    filtered_pairs = [
        (a, b, score)
        for a, b, score in raw_pairs
        if 0.75 <= score < 1.00
    ][:6]

    sim_pairs_str = "\n".join(
        f"  '{a}' e '{b}' (punteggio {score:.2f})"
        for a, b, score in filtered_pairs
    ) or "  nessuna"

    # Format the log of duplicate sentence structural components already removed
    deleted_str = ""

    for index, item in enumerate(deleted_pairs, 1):
        deleted_str += f"\n  Coppia {index} (similarità {item['similarity']}):\n"
        deleted_str += f"    Mantenuta: {item['kept']}\n"
        deleted_str += f"    Rimossa come duplicato: {item['removed']}\n"

    deleted_str = deleted_str or "  nessuna"

   # Format the log of moderate-similarity sentences recommended for potential combining
    merge_candidate_str = ""

    for index, item in enumerate(merge_candidates, 1):
        merge_candidate_str += f"\n  Coppia {index} (similarità {item['similarity']}):\n"
        merge_candidate_str += f"    A: {item['sentence_1']}\n"
        merge_candidate_str += f"    B: {item['sentence_2']}\n"
        merge_candidate_str += f"    Azione: {item['action']}\n"

    merge_candidate_str = merge_candidate_str or "  nessuna"

    return f"""Stile di riscrittura:
{style}

ANALISI USATA PRIMA DELLA RISCRITTURA:

1. Parole ancora ripetute:
   {repeated_str}

2. Pleonasmi da rimuovere:
   {pleonasm_str}

3. Parole simili o quasi sinonimi:
{sim_pairs_str}

4. Frasi duplicate già rimosse:
{deleted_str}

5. Frasi correlate:
{merge_candidate_str}

REGOLE:
- Usa l'analisi sopra per guidare la riscrittura.
- Se una frase duplicata è stata rimossa, non reinserirla.
- Se sono presenti frasi correlate, uniscile solo quando migliora la chiarezza.
- Rimuovi parole e idee ripetute.
- Rimuovi i pleonasmi.
- Usa un italiano semplice e naturale.
- Non rendere il testo troppo elegante o artificioso.
- Non usare sinonimi difficili solo per variare.
- Mantieni le informazioni importanti.
- Non aggiungere nuove informazioni.
- Restituisci solo il testo finale riscritto.

TESTO:
{pre_merged_text.strip()}"""


class TextRewriter:
    """
    High-level orchestrator class. It coordinates the initial local sentence 
    pre-merges, creates contextual prompts, and manages the execution via Ollama.
    """
    def __init__(
        self,
        model="llama3.1",
        delete_threshold=0.96,
        merge_threshold=0.88,
    ):
        self.model = model
        self.merger = PreMerger(
            delete_threshold=delete_threshold,
            merge_threshold=merge_threshold,
        )

    def rewrite(
        self,
        text,
        repetition_analysis,
        redundancy_report,
        mode="standard",
    ):
        """
        Executes the text optimization pipeline. Pre-cleans input text based on reports, 
        generates instructions, calls the local LLM model, and extracts the post-processed text.
        """
        # Step 1: Run the analytical pre-merger adjustments locally
        (
            pre_merged,
            resolved_repeats,
            merge_candidates,
            deleted_pairs,
        ) = self.merger.merge(
            text=text,
            redundant_sentences=redundancy_report.get(
                "redundant_sentences",
                [],
            ),
            repeated_words=repetition_analysis.get(
                "repeated_words",
                {},
            ),
        )

        # Step 2: Compile the structured instructions prompt
        prompt = build_prompt(
            pre_merged_text=pre_merged,
            repetition_analysis=repetition_analysis,
            redundancy_report=redundancy_report,
            resolved_repeats=resolved_repeats,
            merge_candidates=merge_candidates,
            deleted_pairs=deleted_pairs,
            mode=mode,
        )

       # Step 3: Send system rules and generation prompt to Ollama
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                # Low temperature value ensures deterministic, rule-bound, and stable output variations
                "temperature": 0.1,
            },
        )

        # Step 4: Isolate raw message payload contents
        raw_output = response["message"]["content"]

        # Step 5: Post-process the response to strip conversational filler or metadata logs
        return clean_llm_output(raw_output)