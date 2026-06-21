# Import the language_tool_python library for grammar and spell checking capabilities
import language_tool_python


class GrammarEvaluator:
    def __init__(self):
        # Initialize the LanguageTool instance configured for the Italian language ("it")
        self.tool = language_tool_python.LanguageTool("it")

    def count_errors(self, text):
        """
        Helper method that checks the input text for grammatical/spelling issues
        and returns the total count of detected errors.
        """
        return len(self.tool.check(text))

    def language_tool_score(self, original, corrected):
        """
        Calculates a improvement score based on how many errors were fixed.
        It measures the percentage reduction of errors from the original text to the corrected text.
        """
        # Count errors in both the original and corrected text versions
        original_errors = self.count_errors(original)
        corrected_errors = self.count_errors(corrected)

        # If the original text had no errors to begin with, return a perfect score of 100.0
        if original_errors == 0:
            return 100.0

        # Calculate the proportion of errors that were successfully removed
        reduction = (original_errors - corrected_errors) / original_errors
        
        # Convert the reduction to a percentage and round it to 2 decimal places
        return round(reduction * 100, 2)

    def evaluate(self, original, corrected):
        """
        Main evaluation method that returns a structured dictionary containing
        the baseline error counts and the calculated improvement score.
        """
        return {
            "original_errors": self.count_errors(original),
            "corrected_errors": self.count_errors(corrected),
            "language_tool_score": self.language_tool_score(original, corrected),
        }