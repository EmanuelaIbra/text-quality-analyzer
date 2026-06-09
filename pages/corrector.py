import torch
from gramformer import Gramformer
from happytransformer import HappyTextToText, TTSettings
import re
from pages.repetition_analyzer import RepetitionAnalyzer


class GrammarCorrector:

    def __init__(self, use_gpu=False):

        self.use_gpu = use_gpu and torch.cuda.is_available()

        self._set_seed(1212)

        # Gramformer
        self.gf = Gramformer(
            models=1,
            use_gpu=self.use_gpu
        )

        # HappyTransformer model
        self.happy_tt = HappyTextToText(
            "T5",
            "vennify/t5-base-grammar-correction"
        )

        # Generation settings
        self.settings = TTSettings(
            do_sample=True,
            top_k=50,
            temperature=0.7,
            min_length=1,
            
        )
        self.repetition_analyzer = RepetitionAnalyzer()
        

    def _set_seed(self, seed):

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def gramformer_correct(self, text):

        corrections = self.gf.correct(
            text,
            max_candidates=1
        )

        if not corrections:
            return text

        return list(corrections)[0]

    def happy_transformer_correct(self, text):

        prompt = f"grammar: {text}"

        result = self.happy_tt.generate_text(
            prompt,
            args=self.settings
        )

        return result.text

    def split_into_sentences(self, text):

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text.strip()
        )

        return [
            s.strip()
            for s in sentences
            if s.strip()
        ]

    def correct_text(self, text):

        sentences = self.split_into_sentences(text)

        gramformer_sentences = []
        final_sentences = []
        highlighted_sentences = []

        for sentence in sentences:

            gramformer_output = self.gramformer_correct(
                sentence
            )

            final_output = self.happy_transformer_correct(
                gramformer_output
            )

            highlighted = self.gf.highlight(
                sentence,
                final_output
            )

            gramformer_sentences.append(
                gramformer_output
            )

            final_sentences.append(
                final_output
            )

            highlighted_sentences.append(
                highlighted
            )
        corrected_text = " ".join(final_sentences)

        polished_text = self.repetition_analyzer.clean(
             corrected_text
            )

        return {
            "original": text,

            "gramformer_output":
                " ".join(gramformer_sentences),

            

            "corrected": corrected_text,

            "polished": polished_text,

            "highlighted":
                " ".join(highlighted_sentences)
        }