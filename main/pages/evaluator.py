"""
Italian Grammar Evaluator
--------------------------
Evaluates correction quality using metrics from all three layers:

  - LanguageTool error count (grammar / punctuation)
  - pyspellchecker unknown-word count (spelling)
  - A combined weighted score

Install:
    pip install language_tool_python pyspellchecker
"""

import language_tool_python
from spellchecker import SpellChecker
import re


class GrammarEvaluator:
    """
    Evaluates Italian text quality using LanguageTool (grammar) and
    pyspellchecker (spelling) without any AI dependency.

    Parameters
    ----------
    lt_weight    : float  Weight of LanguageTool score in the final score (0–1).
    spell_weight : float  Weight of spell-checker score in the final score (0–1).
                          lt_weight + spell_weight should equal 1.0.
    """

    def __init__(self, lt_weight: float = 0.6, spell_weight: float = 0.4):
        if abs(lt_weight + spell_weight - 1.0) > 1e-6:
            raise ValueError("lt_weight + spell_weight must equal 1.0")

        self._lt          = language_tool_python.LanguageTool("it")
        self._spell       = SpellChecker(language="it")
        self._lt_weight   = lt_weight
        self._spell_weight = spell_weight

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lt_error_count(self, text: str) -> int:
        """Count LanguageTool grammar/punctuation issues."""
        return len(self._lt.check(text))

    def _spell_error_count(self, text: str) -> int:
        """Count words unknown to the Italian pyspellchecker dictionary."""
        words = re.findall(r"\b[A-Za-zÀ-ÿ]+\b", text.lower())
        return len(self._spell.unknown(words))

    @staticmethod
    def _improvement_score(before: int, after: int) -> float:
        """
        Percentage of errors removed.
        Returns 100.0 if there were no errors to begin with.
        Returns 0.0  if no errors were fixed (or new ones introduced).
        """
        if before == 0:
            return 100.0
        reduction = (before - after) / before
        return round(max(reduction, 0.0) * 100, 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, original: str, corrected: str) -> dict:
        """
        Compare original and corrected text across both checking layers.

        Returns
        -------
        dict with keys:
            lt_errors_original      – LanguageTool error count in original
            lt_errors_corrected     – LanguageTool error count in corrected
            lt_score                – % of grammar errors fixed (0–100)

            spell_errors_original   – misspelled words in original
            spell_errors_corrected  – misspelled words in corrected
            spell_score             – % of spelling errors fixed (0–100)

            combined_score          – weighted average of lt_score & spell_score
        """
        lt_orig    = self._lt_error_count(original)
        lt_corr    = self._lt_error_count(corrected)
        spell_orig = self._spell_error_count(original)
        spell_corr = self._spell_error_count(corrected)

        lt_score    = self._improvement_score(lt_orig,    lt_corr)
        spell_score = self._improvement_score(spell_orig, spell_corr)

        combined = round(
            self._lt_weight * lt_score + self._spell_weight * spell_score, 2
        )

        return {
            # Grammar layer
            "lt_errors_original":    lt_orig,
            "lt_errors_corrected":   lt_corr,
            "lt_score":              lt_score,

            # Spelling layer
            "spell_errors_original":  spell_orig,
            "spell_errors_corrected": spell_corr,
            "spell_score":            spell_score,

            # Combined
            "combined_score": combined,
        }