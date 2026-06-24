import re
import ollama


def split_sentences(text: str) -> list[str]:
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text.strip())
        if s.strip()
    ]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


class PreMerger:
    """
    Pre-processes redundant sentence pairs before LLM rewriting.

    - Sentences >= user_choice_threshold: returned as user_choice_candidates
      (LLM keeps both unless the user has already decided via apply_user_decisions)
    - Sentences in [merge_threshold, user_choice_threshold): returned as
      merge_candidates for the LLM to attempt merging
    """

    def __init__(
        self,
        nlp=None,
        user_choice_threshold: float = 0.85,
        merge_threshold:       float = 0.82,
    ):
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
        # Normalise the working text and build a lookup for fast membership tests
        norm_text      = _normalize(text)
        sentences      = split_sentences(norm_text)
        sentence_set   = {_normalize(s) for s in sentences}

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
        remaining_text = " ".join(remaining).lower()
        return {
            word
            for word in repeated_words
            if len(re.findall(rf"\b{re.escape(word)}\b", remaining_text)) <= 1
        }


# ---------------------------------------------------------------------------
# LLM output cleaning
# ---------------------------------------------------------------------------

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


def clean_llm_output(text: str) -> str:
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
    style = _STYLE_MAP.get(mode, _STYLE_MAP["standard"])

    repeated_raw  = repetition_analysis.get("repeated_words", {})
    still_repeated = {
        w: c for w, c in (
            repeated_raw.items() if isinstance(repeated_raw, dict)
            else {w: 2 for item in repeated_raw for w in item.get("words", [])}.items()
        )
        if c >= 2 and w not in resolved_repeats
    }
    repeated_str = ", ".join(
        f"'{w}' (x{c})" for w, c in still_repeated.items()
    ) or "nessuna"

    pleonasm_str = "\n".join(
        f"  '{p['phrase']}' → '{p['replacement']}'"
        for p in redundancy_report.get("pleonasms", [])
    ) or "  nessuno"

    sim_pairs_str = "\n".join(
        f"  '{a}' e '{b}' (punteggio {s:.2f})"
        for a, b, s in redundancy_report.get("similar_words", [])
        if 0.75 <= s < 1.00
    )[:6 * 60] or "  nessuna"       # rough cap on prompt length

    deleted_str = "  nessuna"
    if deleted_pairs:
        deleted_str = ""
        for idx, item in enumerate(deleted_pairs, 1):
            deleted_str += (
                f"\n  Coppia {idx} (similarità {item['similarity']}):\n"
                f"    Mantenuta: {item['kept']}\n"
                f"    Rimossa: {item['removed']}\n"
            )

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
- Le frasi scelte dall'utente sono vincolanti: devono comparire nel testo finale senza modifiche.
- Non modificare, eliminare o spostare i placeholder nel formato [[KEEP_SENTENCE_X]].

TESTO:
{pre_merged_text.strip()}"""


# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------

class TextRewriter:
    def __init__(
        self,
        model:                 str   = "llama3.1",
        nlp                         = None,
        user_choice_threshold: float = 0.90,
        merge_threshold:       float = 0.82,
    ):
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
        mode:                str = "standard",
    ) -> str:
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