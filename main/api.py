from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional

from writing_service import WritingService


app = FastAPI()
service = WritingService()


class AnalyzeRequest(BaseModel):
    text: str


class RewriteRequest(BaseModel):
    text: str
    mode: str = "concise"
    decisions: Optional[Dict[str, str]] = None


@app.get("/")
def home():
    return {
        "message": "Italian Text Quality API is running"
    }


@app.post("/analyze")
def analyze_text_endpoint(request: AnalyzeRequest):
    result = service.analyze_only(
        request.text,
        fast=True,
    )

    return {
        "success": True,
        "original": result["original"],
        "grammar_corrected": result["grammar_corrected"],
        "polished": result["polished"],
        "pleonasm_cleaned": result["pleonasm_cleaned"],

        "grammar_matches": result["grammar_matches"],
        "grammar_metrics": result["grammar_metrics_before_rewrite"],

        "repetition_analysis": result["repetition_analysis"],
        "redundancy_report": result["redundancy_report"],

        "user_choice_candidates": result["user_choice_candidates"],
        "merge_candidates": result["merge_candidates"],
    }


@app.post("/rewrite")
def rewrite_text_endpoint(request: RewriteRequest):
    result = service.rewrite_after_analysis(
        text=request.text,
        mode=request.mode,
        decisions=request.decisions or {},
        final_check=False,
    )

    return {
        "success": True,
        "rewritten": result["rewritten"],
        "final": result["final"],
        "final_grammar_matches": result["final_grammar_matches"],
        "metrics": result["final_metrics"],
    }