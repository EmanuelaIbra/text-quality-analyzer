from fastapi import FastAPI
from pydantic import BaseModel

from gec_project.src.main.writing_service import WritingService


app = FastAPI()
service = WritingService()


class TextRequest(BaseModel):
    text: str
    mode: str = "concise"


@app.get("/")
def home():
    return {
        "message": "Italian Text Quality API is running"
    }


@app.post("/rewrite")
def rewrite_text(request: TextRequest):
    result = service.process(
        request.text,
        mode=request.mode,
        final_check=False,
        fast=True,
        include_full_analysis=False,
    )

    redundancy_report = result.get("redundancy_report", {})

    return {
        "original": result.get("original", request.text),
        "grammar_corrected": result.get("grammar_corrected", ""),
        "polished": result.get("polished", ""),
        "rewritten": result.get("rewritten", ""),
        "final": result.get("final", ""),
        "grammar_matches": result.get("grammar_matches", []),
        "final_grammar_matches": result.get("final_grammar_matches", []),
        "metrics": result.get("final_metrics", {}),
        "repetition_analysis": result.get(
            "repetition_corrected",
            result.get("repetition_analysis", "")
        ),
        "redundancy_report": {
            "pleonasms": redundancy_report.get("pleonasms", []),
            "repeated_words": redundancy_report.get("repeated_words", []),
            "similar_words": redundancy_report.get("similar_words", []),
            "redundant_sentences": redundancy_report.get("redundant_sentences", []),
        },
    }