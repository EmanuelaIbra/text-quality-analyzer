# Text Quality Analyzer & Rewriter

## Overview

 Text Quality Analyzer & Rewriter is an NLP-powered writing assistant designed to improve the quality of English text through grammar correction, repetition detection, redundancy analysis, pleonasm removal, semantic similarity analysis, and AI-assisted rewriting.

The project combines classical NLP techniques, transformer-based language models, semantic embeddings, and local Large Language Models (LLMs) through Ollama to produce clearer, more concise, and higher-quality text while preserving the original meaning.

---

## Features

### Grammar Correction

- Gramformer grammar correction
- HappyTransformer grammar refinement
- Final grammar validation pass

### Repetition Analysis

- Direct word repetition detection
- Repeated sentence detection
- Lexical repetition analysis
- Lemma repetition analysis using spaCy
- Synonym repetition detection using WordNet

### Redundancy Detection

- TF-IDF sentence similarity
- Semantic sentence similarity using Sentence Transformers
- Redundant sentence detection
- Similar word detection
- Semantic overlap analysis

### Pleonasm Detection

Supports:

- Fixed pleonasm dictionary
- SPC dataset lookup

Examples:

| Original | Improved |
|-----------|-----------|
| free gift | gift |
| final outcome | outcome |
| due to the fact that | because |
| at this point in time | now |

### AI Rewriting

Uses Ollama and Llama 3.1 to:

- Merge redundant ideas
- Reduce repetition
- Remove pleonasms
- Improve clarity
- Simplify wording
- Preserve meaning
- Improve flow and readability

### Google Docs Integration

The project can be connected directly to Google Docs using Apps Script.

Features:

- Analyze selected text
- Show repetition analysis
- Show redundancy analysis
- Generate improved version
- Accept or reject changes
- Replace text directly in the document

---

# Pipeline

```text
Input Text
    │
    ▼
GrammarCorrector
    │
    ▼
RepetitionAnalyzer
    │
    ├── Direct repetition
    ├── Lexical repetition
    ├── Lemma repetition
    └── Synonym repetition
    │
    ▼
TextRedundancyChecker
    │
    ├── TF-IDF similarity
    ├── Semantic similarity
    ├── Similar words
    └── Redundant sentences
    │
    ▼
TextRewriter (Ollama + Llama 3.1)
    │
    ▼
Final Grammar Correction
    │
    ▼
Optimized Text
```

---

# Technologies

## NLP

- spaCy
- NLTK
- WordNet
- SentenceTransformers

## Machine Learning

- Transformers
- Gramformer
- HappyTransformer
- PyTorch

## Local LLM

- Ollama
- Llama 3.1

## API

- FastAPI
- Uvicorn

## Integration

- Google Apps Script
- Google Docs

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/gec_project.git

cd gec_project
```

## Install Requirements

```bash
pip install -r requirements.txt
```

## Download spaCy Models

```bash
python -m spacy download en_core_web_sm

python -m spacy download en_core_web_md
```

## Download NLTK Resources

```python
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
```

---

# Ollama Setup

Install Ollama:

https://ollama.com

Pull the model:

```bash
ollama pull llama3.1
```

Verify installation:

```bash
ollama run llama3.1
```

If the model starts successfully, the rewriter is ready.

---

# Running the Console Application

```bash
python app.py
```

Example:

```text
Enter text:

The meeting was extremely important for all the people involved.
It was a very significant moment for the team.
```

Output:

```text
Grammar Analysis
Repetition Analysis
Redundancy Analysis
AI Rewrite
Final Optimized Text
```

---

# Running the API

Start FastAPI:

```bash
uvicorn api:app --reload
```



---

# API Example

Request:

```json
{
  "text": "The meeting was extremely important for all the people involved.",
  "mode": "concise"
}
```

Response:

```json
{
  "original": "...",
  "grammar_corrected": "...",
  "repetition_analysis": {},
  "redundancy_report": {},
  "rewritten": "...",
  "final": "..."
}
```

---

# Google Docs Integration

## Local Development

Expose the API with ngrok:

```bash
ngrok http 8000
```

Example:

```text
https://example.ngrok-free.dev
```

Update Apps Script:

```javascript
const API_URL =
  "https://example.ngrok-free.dev/rewrite";
```

## Workflow

1. Select text in Google Docs
2. Open AI Writing Assistant
3. Analyze text
4. Review analysis
5. Review rewritten version
6. Accept changes
7. Replace selected text

---

# Project Structure

```text
gec_project/

│
├── main.py
├── api.py
│
├── data/
│   └── SPC.jsonl
│
├── pages/
│   ├── corrector.py
│   ├── evaluator.py
│   ├── repetition_analyzer.py
│   ├── text_redundancy_checker.py
│   └── text_rewriter.py
│
├── requirements.txt
└── README.md
```

---

# Example Improvements

### Original

```text
In our advance planning for the future, we need to collaborate together as a team to achieve our future goals.
```

### Optimized

```text
We need to collaborate as a team to achieve our goals.
```

---

### Original

```text
It is critically essential and absolutely necessary that we merge our ideas together.
```

### Optimized

```text
It is essential that we combine our ideas.
```



