"""
text_rewriter.py
================
Ollama-powered text rewriter.

Main idea:
1. Use analysis to safely remove only very-high-confidence duplicate sentences.
2. Keep medium-similarity sentences and ask Ollama to merge them carefully.
3. Use simple, natural English.
4. Preserve useful information.
"""

import re
import spacy
import ollama


try:
    _NLP = spacy.load("en_core_web_md")
except OSError:
    _NLP = None


def split_sentences(text: str) -> list[str]:
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

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "it", "its", "this", "that", "these", "those", "of", "in", "on",
        "at", "to", "for", "with", "by", "from", "and", "or", "but", "if",
        "because", "as", "not", "no", "very", "extremely", "many", "large",
        "number", "several", "some", "have", "has", "had", "will", "would",
        "could", "should", "may", "might", "do", "does", "did", "there",
        "their", "they", "he", "she", "we", "i", "you", "my", "our",
    }

    def __init__(
        self,
        delete_threshold: float = 0.96,
        merge_threshold: float = 0.88,
    ):
        self.delete_threshold = delete_threshold
        self.merge_threshold = merge_threshold

    def merge(
        self,
        text: str,
        redundant_sentences: list,
        repeated_words: dict,
    ) -> tuple[str, set, list, list]:
        sentences = split_sentences(text)
        pairs = sorted(
            redundant_sentences,
            key=lambda item: -item[2]
        )

        dropped = set()
        deleted_pairs = []
        merge_candidates = []

        for sent_a, sent_b, score in pairs:
            if sent_a in dropped or sent_b in dropped:
                continue

            if sent_a not in sentences or sent_b not in sentences:
                continue

            if score >= self.delete_threshold:
                to_drop = self._choose_drop(sent_a, sent_b)
                to_keep = sent_b if to_drop == sent_a else sent_a

                dropped.add(to_drop)

                deleted_pairs.append({
                    "kept": to_keep,
                    "removed": to_drop,
                    "similarity": round(score, 2)
                })

            elif self.merge_threshold <= score < self.delete_threshold:
                merge_candidates.append({
                    "sentence_1": sent_a,
                    "sentence_2": sent_b,
                    "similarity": round(score, 2),
                    "action": (
                        "merge only if it improves clarity; "
                        "do not remove useful information"
                    )
                })

        clean_sentences = [
            sentence
            for sentence in sentences
            if sentence not in dropped
        ]

        resolved_repeats = self._find_resolved_repeats(
            dropped=dropped,
            remaining=clean_sentences,
            repeated_words=repeated_words
        )

        merged_text = " ".join(clean_sentences)

        return (
            merged_text,
            resolved_repeats,
            merge_candidates,
            deleted_pairs
        )

    def _content_words(self, sentence: str) -> set[str]:
        tokens = re.findall(r"[a-z]+", sentence.lower())

        return {
            token
            for token in tokens
            if token not in self.STOPWORDS
        }

    def _choose_drop(self, sent_a: str, sent_b: str) -> str:
        words_a = self._content_words(sent_a)
        words_b = self._content_words(sent_b)

        if words_a <= words_b:
            return sent_a

        if words_b <= words_a:
            return sent_b

        return sent_a if len(words_a) <= len(words_b) else sent_b

    def _find_resolved_repeats(
        self,
        dropped: set,
        remaining: list,
        repeated_words: dict,
    ) -> set:
        remaining_text = " ".join(remaining).lower()
        resolved = set()

        for word in repeated_words:
            count = len(
                re.findall(
                    rf"\b{re.escape(word)}\b",
                    remaining_text
                )
            )

            if count <= 1:
                resolved.add(word)

        return resolved


_PREAMBLE_PATTERNS = [
    r"^here(?:'s| is) (?:the )?rewritten(?: text)?[:\-]?\s*",
    r"^rewritten(?: text)?[:\-]?\s*",
    r"^here(?:'s| is) (?:the )?(?:revised|updated|improved|edited)(?: text)?[:\-]?\s*",
    r"^(?:i )?(?:have )?rewritten(?: the text)?[:\-]?\s*",
    r"^(?:sure[,!]?\s*)?here(?:'s| is)[^:]*:\s*",
]

_POSTAMBLE_MARKERS = [
    r"^i made the following",
    r"^changes(?:\s+i made)?:",
    r"^here(?:'s| are) (?:the )?changes",
    r"^\*\s+merged",
    r"^\*\s+reduced",
    r"^\*\s+removed",
    r"^\*\s+kept",
    r"^\*\s+replaced",
    r"^note[s]?:",
    r"^explanation[s]?:",
    r"^rationale:",
]


def clean_llm_output(text: str) -> str:
    text = text.strip()

    for pattern in _PREAMBLE_PATTERNS:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE
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
    "You are a careful English editor. "
    "Rewrite using clear, simple, natural English. "
    "Do not use fancy, advanced, academic, or dramatic words unless they already appear in the original text. "
    "Preserve useful information. "
    "Do not add new facts. "
    "Return only the rewritten text."
)


def build_prompt(
    pre_merged_text: str,
    repetition_analysis: dict,
    redundancy_report: dict,
    resolved_repeats: set,
    merge_candidates: list,
    deleted_pairs: list,
    mode: str,
) -> str:
    style = {
        "concise": (
            "Make the text shorter and clearer, but do not remove important information."
        ),
        "academic": (
            "Use a clear formal style, but avoid unnecessarily advanced words."
        ),
        "fluent": (
            "Make the text sound natural and smooth, using simple vocabulary."
        ),
        "standard": (
            "Use clear, direct, professional English."
        ),
    }.get(mode, "Use clear, direct, professional English.")

    repeated_raw = repetition_analysis.get("repeated_words", {})

    still_repeated = {
        word: count
        for word, count in repeated_raw.items()
        if count >= 2 and word not in resolved_repeats
    }

    repeated_str = ", ".join(
        f"'{word}' (x{count})"
        for word, count in still_repeated.items()
    ) or "none"

    pleonasm_str = ", ".join(
        f"'{phrase}'"
        for phrase in redundancy_report.get("pleonasms", [])
    ) or "none"

    raw_pairs = redundancy_report.get("similar_words", [])

    filtered_pairs = [
        (a, b, score)
        for a, b, score in raw_pairs
        if 0.75 <= score < 1.00
    ][:6]

    sim_pairs_str = "\n".join(
        f"  '{a}' and '{b}' (score {score:.2f})"
        for a, b, score in filtered_pairs
    ) or "  none"

    deleted_str = ""

    for index, item in enumerate(deleted_pairs, 1):
        deleted_str += f"\n  Pair {index} (similarity {item['similarity']}):\n"
        deleted_str += f"    Kept: {item['kept']}\n"
        deleted_str += f"    Removed duplicate: {item['removed']}\n"

    deleted_str = deleted_str or "  none"

    merge_candidate_str = ""

    for index, item in enumerate(merge_candidates, 1):
        merge_candidate_str += f"\n  Pair {index} (similarity {item['similarity']}):\n"
        merge_candidate_str += f"    A: {item['sentence_1']}\n"
        merge_candidate_str += f"    B: {item['sentence_2']}\n"
        merge_candidate_str += f"    Action: {item['action']}\n"

    merge_candidate_str = merge_candidate_str or "  none"

    return f"""Rewrite style:
{style}

ANALYSIS USED BEFORE REWRITING:

1. Repeated words still present:
   {repeated_str}

2. Pleonasms to remove:
   {pleonasm_str}

3. Similar or near-synonym words:
{sim_pairs_str}

4. Duplicate sentences already removed:
{deleted_str}

5. Related sentence pairs:
{merge_candidate_str}

RULES:
- Use the analysis above to guide the rewrite.
- If a duplicate sentence was already removed, do not bring it back.
- If related sentence pairs are listed, merge them only when it keeps the meaning clear.
- Remove repeated words and repeated ideas.
- Remove pleonasms.
- Use simple, natural vocabulary.
- Do not make the text fancy.
- Do not use advanced synonyms just to sound better.
- Keep important facts.
- Do not add new information.
- Return only the final rewritten text.

TEXT:
{pre_merged_text.strip()}"""


class TextRewriter:

    def __init__(
        self,
        model: str = "llama3.1",
        delete_threshold: float = 0.96,
        merge_threshold: float = 0.88,
    ):
        self.model = model
        self.merger = PreMerger(
            delete_threshold=delete_threshold,
            merge_threshold=merge_threshold
        )

    def rewrite(
        self,
        text: str,
        repetition_analysis: dict,
        redundancy_report: dict,
        mode: str = "standard",
    ) -> str:
        (
            pre_merged,
            resolved_repeats,
            merge_candidates,
            deleted_pairs
        ) = self.merger.merge(
            text=text,
            redundant_sentences=redundancy_report.get(
                "redundant_sentences",
                []
            ),
            repeated_words=repetition_analysis.get(
                "repeated_words",
                {}
            ),
        )

        prompt = build_prompt(
            pre_merged_text=pre_merged,
            repetition_analysis=repetition_analysis,
            redundancy_report=redundancy_report,
            resolved_repeats=resolved_repeats,
            merge_candidates=merge_candidates,
            deleted_pairs=deleted_pairs,
            mode=mode,
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            options={
                "temperature": 0.1
            },
        )

        raw_output = response["message"]["content"]

        return clean_llm_output(raw_output)