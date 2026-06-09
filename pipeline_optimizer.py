import re


class PipelineOptimizer:

    def __init__(
        self,
        corrector,
        text_cleaner,
        text_compressor,
        redundancy_analyzer,
    ):
        self.corrector = corrector
        self.text_cleaner = text_cleaner
        self.text_compressor = text_compressor
        self.redundancy_analyzer = redundancy_analyzer

    def fix_spacing(self, text):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        return text.strip()

    def fix_articles(self, text):
        replacements = {
            "an surprise": "a surprise",
            "a important": "an important",
            "a event": "an event",
            "a outcome": "an outcome",
        }

        for wrong, correct in replacements.items():
            text = re.sub(
                r"\b" + re.escape(wrong) + r"\b",
                correct,
                text,
                flags=re.IGNORECASE,
            )

        return text

    def fix_capitalization(self, text):
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        fixed = []

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence:
                fixed.append(sentence[0].upper() + sentence[1:])

        return " ".join(fixed)

    def postprocess(self, text):
        text = self.fix_spacing(text)
        text = self.fix_articles(text)
        text = self.fix_capitalization(text)
        text = self.fix_spacing(text)
        return text

    def optimize(self, text):
        grammar_result = self.corrector.correct_text(text)
        grammar_text = grammar_result["corrected"]

        repetition_text = grammar_result["polished"]

        cleaner_result = self.text_cleaner.clean(repetition_text)

        pleonasm_text = self.postprocess(
            cleaner_result["pleonasm_text"]
        )

        redundancy_text = self.postprocess(
            cleaner_result["redundancy_text"]
        )

        mmr_text = self.postprocess(
            cleaner_result["mmr_text"]
        )

        semantic_result = self.redundancy_analyzer.analyze(
            mmr_text,
            semantic_threshold=self.text_compressor.similarity_threshold,
        )

        redundant_pairs = semantic_result["semantic_redundant_pairs"]

        compression_result = self.text_compressor.compress(
            mmr_text,
            redundant_pairs=redundant_pairs,
        )

        compressed_text = self.postprocess(
            compression_result["compressed_text"]
        )

        final_result = self.corrector.correct_text(compressed_text)

        final_text = self.postprocess(
            final_result["corrected"]
        )

        return {
            "original": text,
            "grammar_text": grammar_text,
            "repetition_text": repetition_text,
            "pleonasm_text": pleonasm_text,
            "redundancy_text": redundancy_text,
            "mmr_text": mmr_text,
            "semantic_result": semantic_result,
            "compressed_text": compressed_text,
            "final_text": final_text,
            "grammar_result": grammar_result,
            "cleaner_result": cleaner_result,
            "pleonasm_result": cleaner_result["pleonasm_result"],
            "rewrite_result": cleaner_result["rewrite_result"],
            "mmr_result": cleaner_result["mmr_result"],
            "compression_result": compression_result,
            "final_grammar_result": final_result,
        }