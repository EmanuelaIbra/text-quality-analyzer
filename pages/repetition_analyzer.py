import re
import spacy
from nltk.corpus import wordnet as wn
from collections import Counter, defaultdict


class RepetitionAnalyzer:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "is", "are", "am",
            "was", "were", "be", "been", "being", "to", "of", "in",
            "on", "for", "with", "that", "this", "it", "as", "at",
            "by", "from", "my", "your", "his", "her", "their", "our",
            "i", "you", "he", "she", "we", "they", "me", "him", "them"
        }

    # ------------------------
    # Basic tokenization
    # ------------------------

    def tokenize(self, text):
        return re.findall(
            r"\b[a-zA-Z]+\b",
            text.lower()
        )

    def get_content_words(self, text):
        words = self.tokenize(text)

        return [
            word
            for word in words
            if word not in self.stop_words
            and len(word) > 2
        ]

    # ------------------------
    # Direct repetition cleanup
    # ------------------------

    def remove_direct_word_repetition(self, text):
        return re.sub(
            r"\b(\w+)(\s+\1\b)+",
            r"\1",
            text,
            flags=re.IGNORECASE
        )

    def remove_repeated_sentences(self, text):
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text.strip()
        )

        seen = set()
        result = []

        for sentence in sentences:
            normalized = sentence.lower().strip()

            if normalized not in seen:
                seen.add(normalized)
                result.append(sentence)

        return " ".join(result)

    def clean(self, text):
        text = self.remove_direct_word_repetition(text)
        text = self.remove_repeated_sentences(text)
        return text

    # ------------------------
    # Lexical repetition
    # ------------------------

    def repeated_words(self, text):
        words = self.get_content_words(text)
        counts = Counter(words)

        return {
            word: count
            for word, count in counts.items()
            if count > 1
        }

    def top_repeated_words(self, text, limit=10):
        words = self.get_content_words(text)
        counts = Counter(words)

        return [
            (word, count)
            for word, count in counts.most_common(limit)
            if count > 1
        ]

    def lexical_diversity_score(self, text):
        words = self.get_content_words(text)

        if not words:
            return 100.0

        score = len(set(words)) / len(words) * 100

        return round(score, 2)

    def repetition_ratio(self, text):
        words = self.get_content_words(text)

        if not words:
            return 0.0

        repeated = self.repeated_words(text)

        repeated_total = sum(
            count - 1
            for count in repeated.values()
        )

        ratio = repeated_total / len(words) * 100

        return round(ratio, 2)

    def highlight_repetition(self, text):
        repeated = self.repeated_words(text)

        highlighted = text

        for word in repeated.keys():
            pattern = r"\b(" + re.escape(word) + r")\b"

            highlighted = re.sub(
                pattern,
                r"<lexrep>\1</lexrep>",
                highlighted,
                flags=re.IGNORECASE
            )

        return highlighted

    # ------------------------
    # Lemma repetition
    # ------------------------

    def lemma_repetition(self, text):
        doc = self.nlp(text)

        groups = defaultdict(list)

        for token in doc:
            if (
                token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
                and not token.is_stop
                and not token.is_punct
            ):
                lemma = token.lemma_.lower()
                groups[lemma].append(token.text)

        return {
            lemma: words
            for lemma, words in groups.items()
            if len(words) > 1
        }

    # ------------------------
    # Synonym repetition
    # ------------------------

    def get_synonym_key(self, word):
        synsets = wn.synsets(word)

        if not synsets:
            return word

        return synsets[0].lemmas()[0].name().lower()

    def synonym_repetition(self, text):
        doc = self.nlp(text)

        groups = defaultdict(list)

        for token in doc:
            if (
                token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
                and not token.is_stop
                and not token.is_punct
            ):
                lemma = token.lemma_.lower()
                synonym_key = self.get_synonym_key(lemma)

                groups[synonym_key].append(token.text)

        return {
            key: words
            for key, words in groups.items()
            if len(words) > 1
        }

    # ------------------------
    # Full analysis
    # ------------------------

    def analyze(self, text):
        cleaned_text = self.clean(text)

        synonym_repetition = self.synonym_repetition(text)
        lemma_repetition = self.lemma_repetition(text)

        return {
            "cleaned_text": cleaned_text,
            "had_direct_repetition": cleaned_text != text,

            "repeated_words": self.repeated_words(text),
            "top_repeated_words": self.top_repeated_words(text),
            "lexical_diversity_score": self.lexical_diversity_score(text),
            "repetition_ratio": self.repetition_ratio(text),
            "highlighted_repetition": self.highlight_repetition(text),

            "lemma_repetition": lemma_repetition,

            "synonym_repetition": synonym_repetition,
            "has_synonym_repetition": len(synonym_repetition) > 0
        }