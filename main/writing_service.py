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


class WritingService:
    def __init__(self):
        self.nlp_model = spacy.load("it_core_news_lg")

        warmup_pleonasm_cache(self.nlp_model)

        self.corrector = GrammarCorrector()
        self.evaluator = GrammarEvaluator()

        self.repetition_analyzer = RepetitionAnalyzer(
            nlp=self.nlp_model
        )

        self.rewriter = TextRewriter(
            model="llama3.1",
            nlp=self.nlp_model,
            user_choice_threshold=0.90,
            merge_threshold=0.80,
        )

    def analyze_only(self, text, fast=True):
        word_threshold = 0.92 if fast else 0.88
        sent_threshold = 0.80 if fast else 0.75

        grammar_result = self.corrector.correct_text(text)

        original_text = grammar_result["original"]
        grammar_text = grammar_result["corrected"]
        polished_text = grammar_result["polished"]

        grammar_metrics_before_rewrite = self.evaluator.evaluate(
            original_text,
            grammar_text,
        )

        repetition_corrected = self.repetition_analyzer.analyze(
            grammar_text
        )

        redundancy_report = analyze_text(
            grammar_text,
            word_sim_threshold=word_threshold,
            sent_sim_threshold=sent_threshold,
            nlp=self.nlp_model,
            max_similar_tokens=80 if fast else 140,
            sentence_window=3 if fast else 6,
        )

        pleonasm_cleaned_text = apply_pleonasm_replacements(
            grammar_text,
            redundancy_report["pleonasms"],
        )

        if pleonasm_cleaned_text != grammar_text:
            repetition_for_rewrite = self.repetition_analyzer.analyze(
                pleonasm_cleaned_text
            )
        else:
            repetition_for_rewrite = repetition_corrected

        user_choice_candidates = []
        merge_candidates = []

        for index, pair in enumerate(
            redundancy_report.get("redundant_sentences", []),
            start=1,
        ):
            sent_a = pair[0]
            sent_b = pair[1]
            score = pair[2]
            category = pair[3] if len(pair) > 3 else ""

            if score >= 0.80:
                user_choice_candidates.append({
                    "id": str(index),
                    "sentence_1": sent_a,
                    "sentence_2": sent_b,
                    "similarity": score,
                    "category": category,
                })

            elif 0.75 <= score < 0.80:
                merge_candidates.append({
                    "id": str(index),
                    "sentence_1": sent_a,
                    "sentence_2": sent_b,
                    "similarity": score,
                    "category": category,
                })

        return {
            "original": original_text,
            "grammar_corrected": grammar_text,
            "polished": polished_text,
            "pleonasm_cleaned": pleonasm_cleaned_text,
            "grammar_matches": grammar_result["matches"],
            "grammar_metrics_before_rewrite": grammar_metrics_before_rewrite,
            "repetition_analysis": repetition_for_rewrite,
            "redundancy_report": redundancy_report,
            "user_choice_candidates": user_choice_candidates,
            "merge_candidates": merge_candidates,
        }

    def apply_user_decisions(self, text, candidates, decisions):
        updated_text = text

        for candidate in candidates:
            candidate_id = candidate["id"]
            decision = decisions.get(candidate_id, "keep_both")

            sent_1 = candidate["sentence_1"]
            sent_2 = candidate["sentence_2"]

            if decision == "keep_1":
                updated_text = updated_text.replace(sent_2, "")

            elif decision == "keep_2":
                updated_text = updated_text.replace(sent_1, "")

            elif decision == "keep_both":
                continue

        updated_text = " ".join(updated_text.split())
        return updated_text

    def rewrite_after_analysis(
        self,
        text,
        mode="concise",
        decisions=None,
        final_check=False,
    ):
        decisions = decisions or {}

        analysis = self.analyze_only(
            text,
            fast=True,
        )

        text_for_rewrite = self.apply_user_decisions(
            analysis["pleonasm_cleaned"],
            analysis["user_choice_candidates"],
            decisions,
        )

        rewritten_text = self.rewriter.rewrite(
            text=text_for_rewrite,
            repetition_analysis=analysis["repetition_analysis"],
            redundancy_report=analysis["redundancy_report"],
            mode=mode,
        )

        if final_check:
            final_result = self.corrector.correct_text(rewritten_text)
            final_text = final_result["corrected"]
            final_matches = final_result["matches"]
        else:
            final_text = rewritten_text
            final_matches = []

        final_metrics = self.evaluator.evaluate(
            analysis["original"],
            final_text,
        )

        return {
            "rewritten": rewritten_text,
            "final": final_text,
            "final_grammar_matches": final_matches,
            "final_metrics": final_metrics,
        }

    def process(
        self,
        text,
        mode="concise",
        final_check=False,
        fast=True,
        include_full_analysis=False,
    ):
        analysis = self.analyze_only(
            text,
            fast=fast,
        )

        rewrite_result = self.rewrite_after_analysis(
            text=text,
            mode=mode,
            decisions={},
            final_check=final_check,
        )

        return {
            **analysis,
            **rewrite_result,
        }