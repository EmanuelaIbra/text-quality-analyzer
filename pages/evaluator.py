import language_tool_python
from nltk.translate.gleu_score import sentence_gleu


class GrammarEvaluator:

    def __init__(self):

        self.tool = language_tool_python.LanguageTool(
            "en-US"
        )

    def count_errors(self, text):

        matches = self.tool.check(text)

        return len(matches)

    def language_tool_score(
        self,
        original,
        corrected
    ):

        original_errors = self.count_errors(original)
        corrected_errors = self.count_errors(corrected)

        if original_errors == 0:
            return 100.0

        reduction = (
            original_errors - corrected_errors
        ) / original_errors

        return round(reduction * 100, 2)

    def gleu_score(
        self,
        original,
        corrected
    ):

        score = sentence_gleu(
            [corrected.split()],
            original.split()
        )

        return round(score * 100, 2)

    def hybrid_score(
        self,
        original,
        corrected
    ):

        lt = self.language_tool_score(
            original,
            corrected
        )

        gleu = self.gleu_score(
            original,
            corrected
        )

        hybrid = (0.7 * lt) + (0.3 * gleu)

        return round(hybrid, 2)

    def evaluate(
        self,
        original,
        corrected
    ):

        return {
            "original_errors":
                self.count_errors(original),

            "corrected_errors":
                self.count_errors(corrected),

            "language_tool_score":
                self.language_tool_score(
                    original,
                    corrected
                ),

            "gleu_score":
                self.gleu_score(
                    original,
                    corrected
                ),

            "hybrid_score":
                self.hybrid_score(
                    original,
                    corrected
                )
        }