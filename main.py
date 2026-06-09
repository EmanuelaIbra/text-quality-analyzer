from pages.corrector import GrammarCorrector
from pages.evaluator import GrammarEvaluator
from pages.repetition_analyzer import RepetitionAnalyzer
from pages.text_redundancy_checker import analyze_text, print_report
from pages.text_rewriter import TextRewriter

import spacy


def main():
    print("Text Quality Analyzer + Rewriter")
    print("--------------------------------")

    corrector = GrammarCorrector()
    evaluator = GrammarEvaluator()
    repetition_analyzer = RepetitionAnalyzer()
    rewriter = TextRewriter(model="llama3.1")

    nlp_model = spacy.load("en_core_web_md")

    while True:
        text = input("\nEnter text (or exit): ")

        if text.lower() == "exit":
            break

        grammar_result = corrector.correct_text(text)

        original_text = grammar_result["original"]
        grammar_text = grammar_result["corrected"]
        polished_text = grammar_result["polished"]

        grammar_metrics = evaluator.evaluate(
            original_text,
            grammar_text
        )

        repetition_original = repetition_analyzer.analyze(
            original_text
        )

        repetition_corrected = repetition_analyzer.analyze(
            grammar_text
        )

        redundancy_report = analyze_text(
            grammar_text,
            word_sim_threshold=0.88,
            sent_sim_threshold=0.88,
            nlp=nlp_model
        )

        rewritten_text = rewriter.rewrite(
            text=grammar_text,
            repetition_analysis=repetition_corrected,
            redundancy_report=redundancy_report,
            mode="concise"
        )

        final_result = corrector.correct_text(
            rewritten_text
        )

        final_text = final_result["corrected"]

        final_metrics = evaluator.evaluate(
            original_text,
            final_text
        )

        repetition_final = repetition_analyzer.analyze(
            final_text
        )

        print("\nOriginal:")
        print(original_text)

        print("\nAfter Grammar Check:")
        print(grammar_text)

        print("\nAfter Direct Repetition Cleanup:")
        print(polished_text)

        print("\nGrammar Metrics Before Rewrite:")
        for k, v in grammar_metrics.items():
            print(f"  {k}: {v}")

        print("\nRepetition Analysis - Original:")
        for k, v in repetition_original.items():
            print(f"  {k}: {v}")

        print("\nRepetition Analysis - Grammar Corrected:")
        for k, v in repetition_corrected.items():
            print(f"  {k}: {v}")

        print("\nText Redundancy / Similarity Report:")
        print_report(redundancy_report)

        print("\nOllama Rewritten Text:")
        print(rewritten_text)

        print("\nFinal Grammar-Checked Rewrite:")
        print(final_text)

        print("\nFinal Grammar Metrics:")
        for k, v in final_metrics.items():
            print(f"  {k}: {v}")

        print("\nRepetition Analysis - Final:")
        for k, v in repetition_final.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()