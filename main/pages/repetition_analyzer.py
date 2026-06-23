import re
import spacy
from nltk.corpus import wordnet as wn
from collections import Counter, defaultdict


class RepetitionAnalyzer:
    def __init__(self, nlp=None):
        # Load the large pretrained spaCy model for Italian language processing
        self.nlp = nlp or spacy.load("it_core_news_lg")
        # Extract the default set of Italian stop words (e.g., "il", "di", "su")
        self.stop_words = self.nlp.Defaults.stop_words

    def tokenize(self, text):

        """
        Splits text into lowercase word tokens, matching any letter sequences 
        including Italian accented characters (À-ÿ).
        """
        return re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", text.lower())

    def get_content_words(self, text):

        """
        Filters out tokens to isolate meaningful content words. 
        Excludes default stop words and words consisting of 2 characters or fewer.
        """
        return [
            word
            for word in self.tokenize(text)
            if word not in self.stop_words and len(word) > 2
        ]

    def remove_direct_word_repetition(self, text):
        """
        Removes immediate back-to-back duplicate words (e.g., "casa casa" -> "casa") 
        using regular expression capture groups.
        """
        return re.sub(
            r"\b(\w+)(\s+\1\b)+",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )

    def remove_repeated_sentences(self, text):

        """
        Splits text into individual sentences and removes any exact 
        duplicate sentences while preserving their original order.
        """
        # Split text on whitespace following sentence-ending punctuation marks (. ! ?)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        seen = set()
        result = []

        for sentence in sentences:
            # Normalize for case and whitespace to ensure fair comparison
            normalized = sentence.lower().strip()

            if normalized not in seen:
                seen.add(normalized)
                result.append(sentence)

        return " ".join(result)

    def clean(self, text):
        """
        Orchestrates sequential cleaning by stripping away both direct 
        word repetitions and duplicate sentences.
        """
        text = self.remove_direct_word_repetition(text)
        text = self.remove_repeated_sentences(text)
        return text

    def repeated_words(self, text):
        """
        Returns a dictionary mapping recurring content words to their exact total 
        counts, filtering out words that only appeared once.
        """
        counts = Counter(self.get_content_words(text))

        return {
            word: count
            for word, count in counts.items()
            if count > 1
        }

    def top_repeated_words(self, text, limit=10):
        """
        Retrieves up to a designated 'limit' of the most frequently repeated content words, 
        sorted by highest frequency first.
        """
        counts = Counter(self.get_content_words(text))

        return [
            (word, count)
            for word, count in counts.most_common(limit)
            if count > 1
        ]

    def lexical_diversity_score(self, text):
        """
        Computes the Type-Token Ratio (TTR) score of content words as a percentage.
        Higher percentages denote a richer vocabulary selection with fewer repetitions.
        """
        words = self.get_content_words(text)

        if not words:
            return 100.0
        # Unique content words divided by total content words
        return round(len(set(words)) / len(words) * 100, 2)

    def repetition_ratio(self, text):
        """
        Measures what percentage of overall content words represents redundant text 
        (e.g., if a word appears 3 times, 2 of those instances are marked redundant).
        """
        words = self.get_content_words(text)

        if not words:
            return 0.0

        repeated = self.repeated_words(text)
        # Sum the excess counts above 1 for all repeating words
        repeated_total = sum(count - 1 for count in repeated.values())

        return round(repeated_total / len(words) * 100, 2)

    def highlight_repetition(self, text):
        """
        Wraps any repeating content words present in the text with custom 
        XML-like `<lexrep>` tags for downstream visual rendering.
        """
        highlighted = text

        for word in self.repeated_words(text).keys():
            # Construct a regex matching whole words to avoid highlighting substrings
            pattern = r"\b(" + re.escape(word) + r")\b"

            highlighted = re.sub(
                pattern,
                r"<lexrep>\1</lexrep>",
                highlighted,
                flags=re.IGNORECASE,
            )

        return highlighted

    def lemma_repetition(self, text):
        """
        Groups words by their dictionary base form (lemma) using spaCy. 
        Catches grammatical variations of a word (e.g., "gatti" and "gatto" group together).
        """
        doc = self.nlp(text)
        groups = defaultdict(list)

        for token in doc:
            # target major lexical components, ignoring stop words and punctuation
            if (
                token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
                and not token.is_stop
                and not token.is_punct
            ):
                lemma = token.lemma_.lower()
                groups[lemma].append(token.text)

        # Return only the lemma groups containing actual repetitions
        return {
            lemma: words
            for lemma, words in groups.items()
            if len(words) > 1
        }

    def get_synonym_key(self, word):
        """
        Helper method that queries WordNet to extract a baseline canonical Italian 
        synonym name to group conceptually related words together.
        """
        try:
            lemmas = wn.lemmas(word, lang="ita")

            if not lemmas:
                return word
            
            # Retrieve the first structural synset (semantic concept cluster)
            synset = lemmas[0].synset()
            italian_lemmas = synset.lemmas(lang="ita")

            if not italian_lemmas:
                return word
            
            # Pick the primary Italian lemma name from the concept cluster as a key
            return italian_lemmas[0].name().lower()

        except Exception:
            return word

    def synonym_repetition(self, text):
        """
        Identifies conceptual redundancy by grouping content words that share 
        the same fundamental WordNet synonym representation.
        """
        doc = self.nlp(text)
        groups = defaultdict(list)

        for token in doc:
            if (
                token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
                and not token.is_stop
                and not token.is_punct
            ):
                lemma = token.lemma_.lower()
                # Find the universal synonym group identifier for the word's lemma
                synonym_key = self.get_synonym_key(lemma)
                groups[synonym_key].append(token.text)

        # Filter out groups that only have single instances of a concept
        return {
            key: words
            for key, words in groups.items()
            if len(words) > 1
        }

    def analyze(self, text):
        """
        Comprehensive main analysis function that aggregates data from all text cleaning, 
        statistical calculations, word variations, and synonym checks.
        """
        cleaned_text = self.clean(text)
        lemma_repetition = self.lemma_repetition(text)
        synonym_repetition = self.synonym_repetition(text)

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
            "has_synonym_repetition": len(synonym_repetition) > 0,
        }