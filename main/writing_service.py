import spacy

from pages.corrector import GrammarCorrector
from pages.evaluator import GrammarEvaluator
from pages.repetition_analyzer import RepetitionAnalyzer
from pages.text_redundancy_checker import (
    analyze_text,
    apply_pleonasm_replacements,
)
from pages.text_rewriter import TextRewriter


class WritingService:
    """
    Main service for the Italian text-quality pipeline.

    Pipeline:
    1. LanguageTool grammar correction
    2. Repetition analysis
    3. Redundancy + pleonasm analysis
    4. Deterministic pleonasm cleanup
    5. Ollama rewrite guided by analysis
    6. Optional final LanguageTool check
    """

    def __init__(self):
        self.corrector = GrammarCorrector()
        self.evaluator = GrammarEvaluator()
        self.repetition_analyzer = RepetitionAnalyzer()
        self.rewriter = TextRewriter(model="llama3.1")
        self.nlp_model = spacy.load("it_core_news_lg")

    def process(
        self,
        text,
        mode="concise",
        final_check=True,
        fast=False,
        include_full_analysis=True,
    ):
        """
        Process Italian text through the full analysis-guided rewrite pipeline.

        Parameters:
            text: input text
            mode: concise | standard | academic | fluent
            final_check: run final LanguageTool correction after Ollama
            fast: use stricter thresholds for faster processing
            include_full_analysis: include detailed intermediate analysis in output
        """

        # In fast mode, use stricter thresholds.
        # This reduces the number of similar pairs sent to Ollama.
        if fast:
            word_threshold = 0.92
            sent_threshold = 0.92
        else:
            word_threshold = 0.88
            sent_threshold = 0.88

        grammar_result = self.corrector.correct_text(text)

        original_text = grammar_result["original"]
        grammar_text = grammar_result["corrected"]
        polished_text = grammar_result["polished"]

        repetition_corrected = self.repetition_analyzer.analyze(
            grammar_text,
        )

        redundancy_report = analyze_text(
            grammar_text,
            word_sim_threshold=word_threshold,
            sent_sim_threshold=sent_threshold,
            nlp=self.nlp_model,
        )

        pleonasm_cleaned_text = apply_pleonasm_replacements(
            grammar_text,
            redundancy_report["pleonasms"],
        )

        repetition_after_pleonasm = self.repetition_analyzer.analyze(
            pleonasm_cleaned_text,
        )

        rewritten_text = self.rewriter.rewrite(
            text=pleonasm_cleaned_text,
            repetition_analysis=repetition_after_pleonasm,
            redundancy_report=redundancy_report,
            mode=mode,
        )

        if final_check:
            final_result = self.corrector.correct_text(
                rewritten_text,
            )

            final_text = final_result["corrected"]
            final_matches = final_result["matches"]

        else:
            final_result = {
                "corrected": rewritten_text,
                "matches": [],
            }

            final_text = rewritten_text
            final_matches = []

        final_metrics = self.evaluator.evaluate(
            original_text,
            final_text,
        )

        result = {
            "original": original_text,
            "grammar_corrected": grammar_text,
            "polished": polished_text,
            "pleonasm_cleaned": pleonasm_cleaned_text,
            "rewritten": rewritten_text,
            "final": final_text,
            "grammar_matches": grammar_result["matches"],
            "final_grammar_matches": final_matches,
            "final_metrics": final_metrics,
            "redundancy_report": redundancy_report,
            "repetition_analysis": repetition_after_pleonasm,
        }

        if include_full_analysis:
            grammar_metrics = self.evaluator.evaluate(
                original_text,
                grammar_text,
            )

            repetition_original = self.repetition_analyzer.analyze(
                original_text,
            )

            repetition_final = self.repetition_analyzer.analyze(
                final_text,
            )

            result.update({
                "grammar_metrics_before_rewrite": grammar_metrics,
                "repetition_original": repetition_original,
                "repetition_corrected": repetition_corrected,
                "repetition_after_pleonasm": repetition_after_pleonasm,
                "repetition_final": repetition_final,
            })

        return result