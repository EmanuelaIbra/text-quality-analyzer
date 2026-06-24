# Italian Text Quality Analyzer

## Design Requirement Specification Document

DIBRIS – Università di Genova
Softwere Engineering

<div align='right'>
<b>Authors</b><br>
Emanuela Ibra<br>
Yakup Gürer
</div>

---

### REVISION HISTORY

| Version | Date       | Author(s)                           | Notes                                                          |
| ------- | ---------- | ----------------------------------- | -------------------------------------------------------------- |
| 1.0     | 25/06/2026 | Emanuela Ibra <br> Yakup Gürer      | First version of the Design Requirement Specification document |

---

## Table of Content

1. Introduction

   1. Purpose and Scope
   2. Definitions
   3. Document Overview
   4. Bibliography
2. Project Description

   1. Project Introduction
   2. Technologies Used
   3. Assumptions and Constraints
3. System Overview

   1. System Architecture
   2. System Interfaces
   3. System Data
4. Grammar Correction Module
5. Text Analysis Module
6. AI Rewriting Module

---

# 1 Introduction

The Design Requirement Specification (DRS) describes the architecture, components, technologies, interfaces, and internal behavior of the Italian Text Quality Analyzer. The document provides the technical design necessary to understand, maintain, and extend the system.

## 1.1 Purpose and Scope

The purpose of this document is to describe the design of a software system capable of improving Italian text quality through:

* Grammar correction
* Repetition detection
* Synonim detection
* Pleonasm detection
* Redundancy detection
* AI-assisted rewriting
* Google Docs integration

The system is intended for students, researchers, writers, and professionals who need assistance in producing clearer and more concise Italian texts.

## 1.2 Definitions

| Term               | Description                              |
| ------------       | ---------------------------------------- |
| NLP                | Natural Language Processing              |
| LLM                | Large Language Model                     |
| API                | Application Programming Interface        |
| spaCy              | NLP library used for linguistic analysis |
| nltk               | NLP library used lexical analysis        |
| SentenceTransformer| NLP library used for semantic similarity |
| LanguageTool       | Grammar correction engine                |
| Ollama             | Local platform used to execute LLMs      |
| Pleonasm           | Redundant linguistic expression          |
| FastAPI            | Python framework used for REST APIs      |

## 1.3 Document Overview

This document is organized into the following sections:

* Introduction
* Project Description
* System Architecture
* System Modules
* Dynamic Behavior
* Interfaces and Data Flow

## 1.4 Bibliography

* FastAPI Documentation
* spaCy Documentation
* LanguageTool Documentation
* Ollama Documentation
* NLTK Documentation
* Python Documentation

---

# 2 Project Description

## 2.1 Project Introduction

The Italian Text Quality Analyzer is an NLP-based writing assistant designed to automatically improve Italian texts.

The system combines deterministic linguistic analysis with generative AI techniques. Traditional NLP algorithms identify grammatical errors, repeated concepts, pleonasms, and semantic redundancies, while a local Large Language Model generates a cleaner and more concise version of the text.

The system is also integrated with Google Docs through Google Apps Script, allowing users to analyze and rewrite selected text directly inside a document.

## 2.2 Technologies Used

### Programming Language

* Python 3.11
* JavaScript
* Html


### NLP Technologies

* spaCy
* NLTK
* Open Multilingual WordNet

### Grammar Correction

* LanguageTool

### Artificial Intelligence

* Ollama
* Llama 3.1

### Backend

* FastAPI
* Uvicorn

### API Exposure

* ngrok

### Integration

* Google Apps Script
* Google Docs

### Data Storage

*  Json Italian Pleonasm Dictionary

### Supporting Libraries

*  NumPy
*  functools.lru_cache

## 2.3 Assumptions and Constraints

### Assumptions

* Internet access is available for Google Docs integration.
* Ollama is installed locally.
* Italian spaCy models are available.

### Constraints

* Large texts may require additional processing time.
* Rewrite quality depends on the selected language model.
* Semantic similarity calculations require sufficient system memory.

---

# 3 System Overview

The system follows a pipeline architecture.

Input text is processed sequentially through several modules that progressively improve text quality.

## 3.1 System Architecture

The system consists of the following modules:

1. Grammar Correction Module
2. Repetition Analyzer
3. Pleonasm Detector
4. Redundancy Analyzer
5. AI Rewriter
6. FastAPI Service Layer
7. Google Docs Interface

### Architecture Flow

Input Text

↓

LanguageTool Correction

↓

Repetition Analysis

↓

Pleonasm Detection

↓

Redundancy Analysis

↓

Ollama Rewriting

↓

Final Optimized Text

## 3.2 System Interfaces

### REST API

Endpoint:

```text
POST /rewrite
```

Input:

```json
{
  "text": "...",
  "mode": "concise"
}
```

Output:

```json
{
  "final": "...",
  "rewritten": "...",
  "redundancy_report": {}
}
```

### Google Docs Interface

The system communicates with Google Docs through Apps Script.

Users can:

* Select text
* Analyze content
* Review detected issues
* Accept rewritten text

## 3.3 System Data

### 3.3.1 System Inputs

Input data includes:

* Italian text
* Rewrite mode
* User selections from Google Docs

### 3.3.2 System Outputs

Output data includes:

* Corrected text
* Repetition analysis
* Pleonasm analysis
* Redundancy report
* AI-generated rewrite
* Final optimized version

---

# 4 Grammar Correction Module

The Grammar Correction Module is responsible for identifying, analyzing, and correcting grammatical issues in Italian text. It combines LanguageTool-based correction with spaCy linguistic analysis to provide accurate corrections and detailed feedback.

### Main Component

GrammarCorrector

### Responsibilities

* Execute LanguageTool analysis
* Automatically apply grammar  corrections
* Parse and classify detected issues
* Detect noun-adjective and determiner-noun agreement errors using spaCy
* Detect subject-verb agreement errors using spaCy
* Generate corrected and polished text
* Return structured error information for each detected issue


### Inputs

* Raw Italian text

### Processing Flow

* Validate that the input is not empty.
* Run LanguageTool grammar and spelling analysis.
* Generate an automatically corrected version of the text.
* Parse and classify LanguageTool matches.
* Perform noun agreement checks using spaCy.
* Perform subject-verb agreement checks using spaCy.
* Merge all detected issues into a unified result set.


### Outputs

* Corrected text
* Grammar matches
* Error statistics

### External Dependencies

* language_tool_python 
* Italian spaCy model (it_core_news_lg)

---

# 5 Text Analysis Module

The Text Analysis Module evaluates the quality of the corrected text by identifying repetition, pleonasms, and semantic redundancies.

### Components

* RepetitionAnalyzer
* TextRedundancyChecker

### Responsibilities

#### Repetition Detection

Detects:

* Direct word repetition
* Lexical repetition
* Lemma repetition
* Synonym repetition
* Repeated words within the same sentence

Examples:

* casa casa
* gatto, gatti
* nuovo, moderno, recente

#### Pleonasm Detection

Detects:

* Static pleonasms from JSON dictionary
* Lemma-based pleonasm variations
* Linguistic redundancies

Examples:

* risultato finale
* collaborare insieme
* regalo gratuito
* pianificazione futura

#### Redundancy Detection

Detects:

* Similar sentences
* Repeated concepts
* Near-synonym word pairs
* Duplicate sentences
* Semantically redundant sentences
* Merge-candidate sentences

#### Text Quality Analysis

Calculates:

* Lexical diversity score
* Repetition ratio
* Most repeated words
* Text cleanliness indicators

#### Similarity Analysis

Uses:

* spaCy word vectors for similar-word detection
* SentenceTransformer embeddings for sentence similarity analysis
* Italian WordNet for synonym detection

### Inputs

* Corrected Italian text

### Outputs

* Cleaned text
* Repeated words
* Top repeated words
* Lexical diversity score
* Repetition ratio
* Lemma repetitions
* Synonym repetitions
* Pleonasm matches
* Similar word pairs
* Redundant sentence pairs
* Redundancy classifications
* Text quality report
---

# 6 Text Rewrite Module

The Text Rewrite Module rewrites the corrected and analyzed Italian text to improve clarity, fluency, conciseness, and readability.

### Components

* PreMerger
* TextRewriter

### Responsibilities

#### Sentence Pre-Merging

Prepares redundant sentence pairs before rewriting.

Detects:

* Sentences that can be merged
* Sentences that are highly similar
* Sentences that require user choice
* Repeated ideas that should not be deleted automatically

Uses similarity thresholds:

* `user_choice_threshold`
* `merge_threshold`

#### User Choice Handling

Handles very similar sentence pairs carefully.

If two sentences are highly similar, the module does not automatically delete one of them.

Instead, it marks them as user choice candidates, where the user can:

* Keep sentence A
* Keep sentence B
* Keep both sentences
* Merge both sentences without losing information

#### Merge Candidate Detection

Detects sentence pairs that are related but not identical.

These sentences can be merged only when merging improves clarity and does not remove useful information.

Detects:

* Related sentences
* Repeated concepts
* Sentences with partial semantic overlap
* Sentences that can be combined into a clearer sentence

#### Prompt Generation

Builds a structured prompt for the LLM using the previous analysis results.

Uses:

* Repetition analysis
* Pleonasm analysis
* Similar word pairs
* Redundant sentence pairs
* Merge candidates
* User choice candidates
* Selected rewrite mode

The prompt instructs the model to:

* Remove repeated words and ideas
* Remove pleonasms
* Merge related sentences when useful
* Preserve all important information
* Avoid adding new facts
* Use simple and natural Italian
* Return only the rewritten text

#### LLM-Based Rewriting

Uses an Ollama language model to rewrite the text.

The module sends:

* System prompt
* Structured user prompt
* Rewrite mode
* Pre-merged text
* Text analysis results

Supported rewrite modes:

* `concise`
* `academic`
* `fluent`
* `standard`

#### Output Cleaning

Cleans the LLM output before returning the final text.

Removes:

* Introductory phrases
* Explanations
* Notes
* Bullet-point summaries
* Post-rewrite comments

Examples removed:

* Ecco il testo riscritto:
* Versione migliorata:
* Ho corretto...
* Modifiche:
* Spiegazione:

### Inputs

* Corrected Italian text
* Repetition analysis report
* Redundancy analysis report
* Rewrite mode

### Outputs

* Rewritten text
* Cleaner and more fluent text
* Reduced repetitions
* Removed pleonasms
* Merged sentence candidates
* Preserved important information

### External Dependencies

* Ollama
* Regular expressions
* spaCy

### Example

Original:

"The annual meeting was attended by many employees. A large number of employees attended the annual meeting."

Rewritten:

"Many employees attended the annual meeting."

### Output

The resulting text is returned to the service layer and exposed through both FastAPI and Google Docs integration.

---

# Conclusion

The Italian Text Quality Analyzer combines deterministic NLP techniques and local generative AI to provide high-quality text improvement. The modular architecture enables maintainability, scalability, and future extensions such as readability scoring, multilingual support, and additional writing styles.
