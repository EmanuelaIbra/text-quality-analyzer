import re
import spacy
import nltk
from nltk.corpus import wordnet as wn
from collections import Counter, defaultdict

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

def get_italian_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word, lang='ita'):
        for lemma in syn.lemma_names('ita'):
            clean_synonym = lemma.replace('_', ' ').lower()
            synonyms.add(clean_synonym)
    return synonyms

class RepetitionAnalyzer:
    def __init__(self, nlp=None):
        self.nlp = nlp or spacy.load("it_core_news_lg")
        self.stop_words = self.nlp.Defaults.stop_words
        self._SYNONYM_SKIP = {"ieri", "sera", "molto", "anche", "però", "ormai"}

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
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        seen = set()
        result = []
        for sentence in sentences:
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
        return {word: count for word, count in counts.items() if count > 1}

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
        repeated_total = sum(count - 1 for count in repeated.values())
        return round(repeated_total / len(words) * 100, 2)

    def highlight_repetition(self, text):
        """
        Wraps any repeating content words present in the text with custom 
        XML-like `<lexrep>` tags for downstream visual rendering.
        """
        highlighted = text
        for word in self.repeated_words(text).keys():
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
            if (
                token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
                and not token.is_stop
                and not token.is_punct
            ):
                lemma = token.lemma_.lower()
                groups[lemma].append(token.text)
        return {lemma: words for lemma, words in groups.items() if len(words) > 1}

    def get_synonym_key(self, word):
        """
        Helper method that queries WordNet to extract a baseline canonical Italian 
        synonym name to group conceptually related words together.
        """
        try:
            lemmas = wn.lemmas(word, lang="ita")
            if not lemmas:
                return word
            synset = lemmas[0].synset()
            italian_lemmas = synset.lemmas(lang="ita")
            if not italian_lemmas:
                return word
            return italian_lemmas[0].name().lower()
        except Exception:
            return word

    def synonym_repetition(self, text, window: int = None):
 #Detects synonym redundancy scoped to individual sentences (default) or within a sliding token window if `window` is given as an int.

        # Split into sentences on common Italian punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        detected_pairs = []
        already_paired = set()

        for sentence in sentences:
            cleaned = sentence.lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '')
            words = cleaned.split()

            for i in range(len(words)):
                word_a = words[i]

                if len(word_a) <= 3 or word_a in self._SYNONYM_SKIP:
                    continue

                synonyms_of_a = get_italian_synonyms(word_a)

                # Only compare against words later in the SAME sentence
                search_range = range(i + 1, len(words))

                # Optionally also check a cross-sentence window
                if window is not None:
                    # flatten all words and check by absolute index proximity
                    # (handled below outside the sentence loop — skip here)
                    pass

                for j in search_range:
                    word_b = words[j]

                    if word_b in self._SYNONYM_SKIP or len(word_b) <= 3:
                        continue

                    if word_a != word_b and word_b in synonyms_of_a:
                        pair_key = tuple(sorted([word_a, word_b]))
                        if pair_key not in already_paired:
                            already_paired.add(pair_key)
                            detected_pairs.append({
                                "pair": (word_a, word_b),
                                "sentence": sentence.strip(),
                            })

    # Optional: sliding window across sentence boundaries
        if window is not None:
            all_words = text.lower().replace('.', '').replace(',', '').split()
            for i in range(len(all_words)):
                word_a = all_words[i]
                if len(word_a) <= 3 or word_a in self._SYNONYM_SKIP:
                    continue
                synonyms_of_a = get_italian_synonyms(word_a)
                for j in range(i + 1, min(i + window + 1, len(all_words))):
                    word_b = all_words[j]
                    if len(word_b) <= 3 or word_b in self._SYNONYM_SKIP:
                        continue
                    if word_a != word_b and word_b in synonyms_of_a:
                        pair_key = tuple(sorted([word_a, word_b]))
                        if pair_key not in already_paired:
                            already_paired.add(pair_key)
                            detected_pairs.append({
                                "pair": (word_a, word_b),
                                "sentence": f"(cross-sentence window) …{word_a}… …{word_b}…",
                            })

        return detected_pairs

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
            "synonym_repetition": self.synonym_repetition(text),
            "has_synonym_repetition": len(self.synonym_repetition(text)) > 0,
        }