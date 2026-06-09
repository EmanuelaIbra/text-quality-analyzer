from fastapi import FastAPI
from pydantic import BaseModel

from pages.corrector import GrammarCorrector
from pages.evaluator import GrammarEvaluator
from pages.repetition_analyzer import RepetitionAnalyzer
from pages.text_redundancy_checker import analyze_text
from pages.text_rewriter import TextRewriter

import spacy


app = FastAPI()

corrector = GrammarCorrector()
evaluator = GrammarEvaluator()
repetition_analyzer = RepetitionAnalyzer()
rewriter = TextRewriter(model="llama3.1")
nlp_model = spacy.load("en_core_web_md")


class TextRequest(BaseModel):
    text: str
    mode: str = "concise"


@app.get("/")
def home():
    return {"message": "Grammar API is running"}


@app.post("/rewrite")
def rewrite_text(request: TextRequest):
    grammar_result = corrector.correct_text(request.text)

    grammar_text = grammar_result["corrected"]

    repetition_analysis = repetition_analyzer.analyze(grammar_text)

    redundancy_report = analyze_text(
        grammar_text,
        word_sim_threshold=0.88,
        sent_sim_threshold=0.88,
        nlp=nlp_model
    )

    rewritten_text = rewriter.rewrite(
        text=grammar_text,
        repetition_analysis=repetition_analysis,
        redundancy_report=redundancy_report,
        mode=request.mode
    )

    final_result = corrector.correct_text(rewritten_text)

    final_text = final_result["corrected"]

    metrics = evaluator.evaluate(
        request.text,
        final_text
    )

    return {
        "original": request.text,
        "grammar_corrected": grammar_text,
        "rewritten": rewritten_text,
        "final": final_text,
        "metrics": metrics,
        "repetition_analysis": repetition_analysis,
        "redundancy_report": {
            "pleonasms": redundancy_report["pleonasms"],
            "repeated_words": redundancy_report["repeated_words"],
            "similar_words": redundancy_report["similar_words"],
            "redundant_sentences": redundancy_report["redundant_sentences"],
        }
    }