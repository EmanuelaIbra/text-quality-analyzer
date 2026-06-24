import re
import spacy

from pages.corrector import GrammarCorrector
from pages.evaluator import GrammarEvaluator
from pages.repetition_analyzer import RepetitionAnalyzer
from pages.text_redundancy_checker import (
    analyze_text,
    apply_pleonasm_replacements,
    warmup_pleonasm_cache,
)
from pages.text_rewriter import TextRewriter

# ---------------------------------------------------------------------------
# Thresholds — single source of truth shared by analyze_only and the rewriter
# ---------------------------------------------------------------------------

USER_CHOICE_THRESHOLD = 0.85   # >= this  →  user must pick one sentence
MERGE_THRESHOLD       = 0.82   # >= this  →  LLM may merge (raised to match transformer)
FAST_WORD_THRESHOLD   = 0.92
SLOW_WORD_THRESHOLD   = 0.88
FAST_SENT_THRESHOLD   = 0.82   # same as transformer default
SLOW_SENT_THRESHOLD   = 0.82   # transformer is already at its best; no value in lowering


class WritingService:
    def __init__(self):
        self.nlp_model = spacy.load("it_core_news_lg")

        warmup_pleonasm_cache(self.nlp_model)

        self.corrector = GrammarCorrector()
        self.evaluator = GrammarEvaluator()

        self.repetition_analyzer = RepetitionAnalyzer(nlp=self.nlp_model)

        self.rewriter = TextRewriter(
            model="llama3.1",
            nlp=self.nlp_model,
            user_choice_threshold=USER_CHOICE_THRESHOLD,
            merge_threshold=MERGE_THRESHOLD,
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize(s: str) -> str:
        """Collapse whitespace for fuzzy sentence matching."""
        return re.sub(r"\s+", " ", s.strip())

    def _find_in_text(self, text: str, sentence: str) -> int:
        """
        Return the start index of sentence inside text, normalising whitespace.
        Returns -1 if not found.
        """
        needle = self._normalize(sentence)
        haystack = self._normalize(text)
        return haystack.find(needle)

    def _safe_replace(self, text: str, old: str, new: str) -> str:
        """
        Replace old with new, collapsing internal whitespace before matching
        so minor spacing differences don't cause silent failures.
        """
        needle = self._normalize(old)
        pattern = re.escape(needle)
        # Allow any whitespace run between words
        pattern = re.sub(r"\\ ", r"\\s+", pattern)
        return re.sub(pattern, new, self._normalize(text), count=1)

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------

    def analyze_only(self, text: str, fast: bool = True) -> dict:
        word_threshold = FAST_WORD_THRESHOLD if fast else SLOW_WORD_THRESHOLD
        sent_threshold = FAST_SENT_THRESHOLD if fast else SLOW_SENT_THRESHOLD

        grammar_result = self.corrector.correct_text(text)

        original_text = grammar_result["original"]
        grammar_text  = grammar_result["corrected"]
        polished_text = grammar_result["polished"]

        grammar_metrics_before_rewrite = self.evaluator.evaluate(
            original_text,
            grammar_text,
        )

        repetition_corrected = self.repetition_analyzer.analyze(grammar_text)

        redundancy_report = analyze_text(
            grammar_text,
            word_sim_threshold=word_threshold,
            sent_sim_threshold=sent_threshold,
            nlp=self.nlp_model,
            max_similar_tokens=80 if fast else 140,
        )

        pleonasm_cleaned_text = apply_pleonasm_replacements(
            grammar_text,
            redundancy_report["pleonasms"],
        )

        repetition_for_rewrite = (
            self.repetition_analyzer.analyze(pleonasm_cleaned_text)
            if pleonasm_cleaned_text != grammar_text
            else repetition_corrected
        )

        # -------------------------------------------------------------------
        # Classify redundant sentence pairs
        # Thresholds now align with the transformer-based scorer:
        #   >= USER_CHOICE_THRESHOLD  →  user_choice_candidates
        #   >= MERGE_THRESHOLD        →  merge_candidates
        # -------------------------------------------------------------------
        user_choice_candidates = []
        merge_candidates       = []

        for index, pair in enumerate(
            redundancy_report.get("redundant_sentences", []),
            start=1,
        ):
            sent_a   = pair[0]
            sent_b   = pair[1]
            score    = pair[2]
            category = pair[3] if len(pair) > 3 else ""

            if score >= USER_CHOICE_THRESHOLD:
                user_choice_candidates.append({
                    "id":         str(index),
                    "sentence_1": sent_a,
                    "sentence_2": sent_b,
                    "similarity": score,
                    "category":   category,
                })

            elif score >= MERGE_THRESHOLD:
                merge_candidates.append({
                    "id":         str(index),
                    "sentence_1": sent_a,
                    "sentence_2": sent_b,
                    "similarity": score,
                    "category":   category,
                })

        return {
            "original":                      original_text,
            "grammar_corrected":             grammar_text,
            "polished":                      polished_text,
            "pleonasm_cleaned":              pleonasm_cleaned_text,
            "grammar_matches":               grammar_result["matches"],
            "grammar_metrics_before_rewrite": grammar_metrics_before_rewrite,
            "repetition_analysis":           repetition_for_rewrite,
            "redundancy_report":             redundancy_report,
            "user_choice_candidates":        user_choice_candidates,
            "merge_candidates":              merge_candidates,
        }

    # -----------------------------------------------------------------------
    # User decisions
    # -----------------------------------------------------------------------

    def apply_user_decisions(
        self,
        text:       str,
        candidates: list[dict],
        decisions:  dict,
    ) -> tuple[list[dict], str]:
        """
        Returns a segment list instead of a placeholder-polluted string.

        Each segment is either:
          {"type": "protected", "text": "..."}   ← never shown to the LLM
          {"type": "rewrite",   "text": "..."}   ← sent to the LLM

        Also returns the text with the *removed* sentences already erased,
        so the LLM never sees them either.
        """
        working = self._normalize(text)

        # Sentences to fully remove (the user discarded them)
        to_remove: list[str] = []
        # Sentences to protect (the user kept them — LLM must not touch them)
        to_protect: list[str] = []

        for candidate in candidates:
            cid      = candidate["id"]
            decision = decisions.get(cid, "keep_both")

            if decision == "keep_both":
                continue

            s1 = self._normalize(candidate["sentence_1"])
            s2 = self._normalize(candidate["sentence_2"])

            if decision == "keep_1":
                to_protect.append(s1)
                to_remove.append(s2)
            else:  # keep_2
                to_protect.append(s2)
                to_remove.append(s1)

        # Erase removed sentences from working text first
        for sentence in to_remove:
            working = self._safe_replace(working, sentence, "")
        working = re.sub(r"\s{2,}", " ", working).strip()

        if not to_protect:
            # Nothing protected — send everything to the LLM as one block
            return [{"type": "rewrite", "text": working}], working

        # Split working text into protected / rewrite segments
        # Strategy: find each protected sentence in the text and split around it
        segments: list[dict] = []
        remaining = working

        for protected in to_protect:
            pattern = re.escape(protected)
            pattern = re.sub(r"\\ ", r"\\s+", pattern)
            m = re.search(pattern, remaining, flags=re.IGNORECASE)

            if not m:
                # Can't find it — just keep the whole remaining for rewriting
                # (better to over-rewrite than to silently drop a protected sentence)
                continue

            before = remaining[:m.start()].strip()
            after  = remaining[m.end():].strip()

            if before:
                segments.append({"type": "rewrite",   "text": before})
            segments.append(    {"type": "protected",  "text": protected})
            remaining = after

        if remaining.strip():
            segments.append({"type": "rewrite", "text": remaining.strip()})

        return segments, working

    def _rewrite_segments(
        self,
        segments:            list[dict],
        repetition_analysis: dict,
        redundancy_report:   dict,
        mode:                str,
    ) -> str:
        """
        Rewrite only the 'rewrite' segments; stitch protected ones back in order.
        Protected sentences are never seen by the LLM.
        """
        result_parts: list[str] = []

        for seg in segments:
            if seg["type"] == "protected":
                result_parts.append(seg["text"])
            else:
                chunk = seg["text"].strip()
                if not chunk:
                    continue
                rewritten = self.rewriter.rewrite(
                    text=chunk,
                    repetition_analysis=repetition_analysis,
                    redundancy_report=redundancy_report,
                    mode=mode,
                )
                result_parts.append(rewritten.strip())

        return re.sub(r"\s{2,}", " ", " ".join(result_parts)).strip()

    # -----------------------------------------------------------------------
    # Rewrite
    # -----------------------------------------------------------------------

    def rewrite_after_analysis(
        self,
        text:        str,
        mode:        str  = "concise",
        decisions:   dict = None,
        final_check: bool = False,
    ) -> dict:
        decisions = decisions or {}

        analysis = self.analyze_only(text, fast=True)

        segments, _ = self.apply_user_decisions(
            analysis["pleonasm_cleaned"],
            analysis["user_choice_candidates"],
            decisions,
        )

        # If there are no protected segments, one LLM call covers everything.
        # If there are, each rewrite chunk is called separately and protected
        # sentences are stitched in without ever touching the LLM.
        rewritten_text = self._rewrite_segments(
            segments=segments,
            repetition_analysis=analysis["repetition_analysis"],
            redundancy_report=analysis["redundancy_report"],
            mode=mode,
        )

        if final_check:
            final_result  = self.corrector.correct_text(rewritten_text)
            final_text    = final_result["corrected"]
            final_matches = final_result["matches"]
        else:
            final_text    = rewritten_text
            final_matches = []

        final_metrics = self.evaluator.evaluate(
            analysis["original"],
            final_text,
        )

        return {
            "rewritten":             rewritten_text,
            "final":                 final_text,
            "final_grammar_matches": final_matches,
            "final_metrics":         final_metrics,
        }

    # -----------------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------------

    def process(
        self,
        text:                 str,
        mode:                 str  = "concise",
        final_check:          bool = False,
        fast:                 bool = True,
        include_full_analysis: bool = False,
    ) -> dict:
        analysis       = self.analyze_only(text, fast=fast)
        rewrite_result = self.rewrite_after_analysis(
            text=text,
            mode=mode,
            decisions={},
            final_check=final_check,
        )
        return {**analysis, **rewrite_result}