import re
import ollama


def split_sentences(text):
    """
    Split text into individual sentences.
    """
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


class PreMerger:
    """
    Pre-processes redundant sentence pairs before LLM rewriting.

    Updated behavior:
    - It does NOT automatically delete sentences.
    - Very similar sentences are returned as user-choice candidates.
    - Moderately similar sentences are returned as merge candidates.
    """

    def __init__(
        self,
        nlp=None,
        user_choice_threshold=0.80,
        merge_threshold=0.70,
    ):
        self.nlp = nlp
        self.user_choice_threshold = user_choice_threshold
        self.merge_threshold = merge_threshold

        if self.nlp:
            self.stop_words = self.nlp.Defaults.stop_words
        else:
            self.stop_words = set()

    def merge(self, text, redundant_sentences, repeated_words):
        """
        Analyze redundant sentence pairs without deleting them automatically.
        """
        sentences = split_sentences(text)

        pairs = sorted(
            redundant_sentences,
            key=lambda item: -item[2],
        )

        merge_candidates = []
        user_choice_candidates = []
        deleted_pairs = []
        dropped = set()

        for pair in pairs:
            sent_a = pair[0]
            sent_b = pair[1]
            score = pair[2]
            category = pair[3] if len(pair) > 3 else ""

            if sent_a not in sentences or sent_b not in sentences:
                continue

            if score >= self.user_choice_threshold:
                user_choice_candidates.append({
                    "sentence_1": sent_a,
                    "sentence_2": sent_b,
                    "similarity": round(score, 2),
                    "action": (
                        "scelta_utente: scegliere la frase A, "
                        "la frase B oppure mantenere entrambe"
                    ),
                })

            elif self.merge_threshold <= score < self.user_choice_threshold:
                merge_candidates.append({
                    "sentence_1": sent_a,
                    "sentence_2": sent_b,
                    "similarity": round(score, 2),
                    "action": (
                        "unire solo se migliora la chiarezza; "
                        "non eliminare informazioni utili"
                    ),
                })

        clean_sentences = [
            sentence
            for sentence in sentences
            if sentence not in dropped
        ]

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
            user_choice_candidates,
        )

    def _content_words(self, sentence):
        """
        Extract meaningful content lemmas from a sentence.
        """
        if self.nlp:
            doc = self.nlp(sentence)

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

    def _find_resolved_repeats(self, dropped, remaining, repeated_words):
        """
        Recalculate which repetitions were resolved.
        Since this version does not delete automatically,
        this usually returns an empty set.
        """
        remaining_text = " ".join(remaining).lower()
        resolved = set()

        for word in repeated_words:
            count = len(
                re.findall(
                    rf"\b{re.escape(word)}\b",
                    remaining_text,
                )
            )

            if count <= 1:
                resolved.add(word)

        return resolved


_PREAMBLE_PATTERNS = [
    r"^ecco(?: il)? testo riscritto[:\-]?\s*",
    r"^testo riscritto[:\-]?\s*",
    r"^versione corretta[:\-]?\s*",
    r"^versione migliorata[:\-]?\s*",
    r"^ho riscritto il testo[:\-]?\s*",
    r"^certo[,!]?\s*ecco[^:]*:\s*",
]


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
    Remove conversational preambles, explanations, and bullet lists.
    """
    text = text.strip()

    for pattern in _PREAMBLE_PATTERNS:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        )

    lines = text.strip().splitlines()
    clean_lines = []

    for line in lines:
        stripped = line.strip()

        if any(
            re.match(pattern, stripped, re.IGNORECASE)
            for pattern in _POSTAMBLE_MARKERS
        ):
            break

        clean_lines.append(line)

    text = "\n".join(clean_lines).strip()
    text = re.sub(r"^\s*[\*\-]\s+.+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


SYSTEM_PROMPT = (
    "Sei un revisore professionale di testi italiani. "
    "Riscrivi usando un italiano chiaro, naturale e semplice. "
    "Non usare parole troppo complesse, accademiche o artificiose se non sono già presenti nel testo originale. "
    "Conserva tutte le informazioni importanti. "
    "Non aggiungere nuovi fatti. "
    "Non eliminare informazioni quando non sei sicuro. "
    "Restituisci solo il testo riscritto."
)


def build_prompt(
    pre_merged_text,
    repetition_analysis,
    redundancy_report,
    resolved_repeats,
    merge_candidates,
    deleted_pairs,
    user_choice_candidates,
    mode,
):
    """
    Build the prompt for Ollama using the analysis results.
    """

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

    filtered_pairs = [
        (a, b, score)
        for a, b, score in raw_pairs
        if 0.75 <= score < 1.00
    ][:6]

    sim_pairs_str = "\n".join(
        f"  '{a}' e '{b}' (punteggio {score:.2f})"
        for a, b, score in filtered_pairs
    ) or "  nessuna"

    deleted_str = ""

    for index, item in enumerate(deleted_pairs, 1):
        deleted_str += f"\n  Coppia {index} (similarità {item['similarity']}):\n"
        deleted_str += f"    Mantenuta: {item['kept']}\n"
        deleted_str += f"    Rimossa come duplicato: {item['removed']}\n"

    deleted_str = deleted_str or "  nessuna"

    merge_candidate_str = ""

    for index, item in enumerate(merge_candidates, 1):
        merge_candidate_str += f"\n  Coppia {index} (similarità {item['similarity']}):\n"
        merge_candidate_str += f"    A: {item['sentence_1']}\n"
        merge_candidate_str += f"    B: {item['sentence_2']}\n"
        merge_candidate_str += f"    Azione: {item['action']}\n"

    merge_candidate_str = merge_candidate_str or "  nessuna"

    user_choice_str = ""

    for index, item in enumerate(user_choice_candidates, 1):
        user_choice_str += f"\n  Coppia {index} (similarità {item['similarity']}):\n"
        user_choice_str += f"    A: {item['sentence_1']}\n"
        user_choice_str += f"    B: {item['sentence_2']}\n"
        user_choice_str += (
            "    Azione: non eliminare automaticamente; "
            "se non esiste una scelta esplicita dell'utente, mantieni entrambe "
            "oppure fondile senza perdere informazioni.\n"
        )

    user_choice_str = user_choice_str or "  nessuna"

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
{merge_candidate_str}

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

TESTO:
{pre_merged_text.strip()}"""


class TextRewriter:
    """
    Coordinates pre-merge analysis, prompt generation, and Ollama rewriting.
    """

    def __init__(
        self,
        model="llama3.1",
        nlp=None,
        user_choice_threshold=0.90,
        merge_threshold=0.80,
    ):
        self.model = model
        self.merger = PreMerger(
            nlp=nlp,
            user_choice_threshold=user_choice_threshold,
            merge_threshold=merge_threshold,
        )

    def rewrite(
        self,
        text,
        repetition_analysis,
        redundancy_report,
        mode="standard",
    ):
        (
            pre_merged,
            resolved_repeats,
            merge_candidates,
            deleted_pairs,
            user_choice_candidates,
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
                "temperature": 0.1,
            },
        )

        raw_output = response["message"]["content"]

        return clean_llm_output(raw_output)