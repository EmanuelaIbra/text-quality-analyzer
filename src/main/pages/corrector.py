# Import the language_tool_python library to enable grammar and spell checking
import language_tool_python


class GrammarCorrector:
    def __init__(self):
        # Initialize the LanguageTool instance set specifically for the Italian language ("it")
        self.tool = language_tool_python.LanguageTool("it")

    def get_match_value(self, match, *names, default=None):
        """
        Helper method to safely extract attributes from a match object.
        It loops through a list of potential attribute names (to handle API version differences)
        and returns the first one that exists. If none match, it returns the default value.
        """
        for name in names:
            if hasattr(match, name):
                return getattr(match, name)
        return default

    def correct_text(self, text):
        """
        Analyzes the input text for grammatical errors, generates a corrected version,
        and extracts structured metadata for each identified issue.
        """
        # Find all grammatical issues and mismatches in the input text
        matches = self.tool.check(text)

        # Automatically generate the fully corrected text
        corrected_text = self.tool.correct(text)

        # Return a structured dictionary containing the original text, the corrections,
        # and a detailed breakdown of each found error

        return {
            "original": text,
            "corrected": corrected_text,
            "polished": corrected_text,
            "matches": [
                {   
                    # Safely retrieve the explanation message for the error
                    "message": self.get_match_value(
                        match,
                        "message",
                        default=""
                    ),
                    # Safely retrieve the rule identifier (checks for 'ruleId' or 'rule_id')
                    "rule": self.get_match_value(
                        match,
                        "ruleId",
                        "rule_id",
                        default=""
                    ),

                    # Safely retrieve the starting character position of the error
                    "offset": self.get_match_value(
                        match,
                        "offset",
                        default=0
                    ),

                    # Safely retrieve the length of the error (checks for 'errorLength' or 'error_length')
                    "length": self.get_match_value(
                        match,
                        "errorLength",
                        "error_length",
                        default=0
                    ),

                    # Safely retrieve the replacement suggestions, slicing to limit the output to the top 5
                    "suggestions": self.get_match_value(
                        match,
                        "replacements",
                        default=[]
                    )[:5],
                }
                # List comprehension to iterate through and parse every match found by LanguageTool
                for match in matches
            ],
        }