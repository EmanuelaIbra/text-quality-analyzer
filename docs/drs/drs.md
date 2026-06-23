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
| 1.0     | 09/06/2026 | Emanuela Ibra <br> Yakup Gürer      | First version of the Design Requirement Specification document |

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
* Pleonasm detection
* Semantic similarity analysis
* Redundancy detection
* AI-assisted rewriting
* Google Docs integration

The system is intended for students, researchers, writers, and professionals who need assistance in producing clearer and more concise Italian texts.

## 1.2 Definitions

| Term         | Description                              |
| ------------ | ---------------------------------------- |
| NLP          | Natural Language Processing              |
| LLM          | Large Language Model                     |
| API          | Application Programming Interface        |
| spaCy        | NLP library used for linguistic analysis |
| LanguageTool | Grammar correction engine                |
| Ollama       | Local platform used to execute LLMs      |
| Pleonasm     | Redundant linguistic expression          |
| FastAPI      | Python framework used for REST APIs      |

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

### Integration

* Google Apps Script
* Google Docs

### Data Storage

* JSON files
* Italian Pleonasm Dictionary

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

The Grammar Correction Module is responsible for identifying and correcting grammatical, orthographic, and syntactic errors.

### Main Component

GrammarCorrector

### Responsibilities

* Execute LanguageTool analysis
* Collect grammar suggestions
* Produce corrected text
* Return detailed error information

### Inputs

* Raw Italian text

### Outputs

* Corrected text
* Grammar matches
* Error statistics

---

# 5 Text Analysis Module

The Text Analysis Module evaluates the quality of the corrected text.

### Components

* RepetitionAnalyzer
* TextRedundancyChecker

### Responsibilities

#### Repetition Detection

Detects:

* Lexical repetition
* Lemma repetition
* Synonym repetition

#### Pleonasm Detection

Detects:

* Static pleonasms from JSON dictionary
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

---

# 6 AI Rewriting Module

The AI Rewriting Module generates an improved version of the text according to the analysis.

### Main Component

TextRewriter

### Technology

* Ollama
* Llama 3.1

### Responsibilities

The module receives:

* Corrected text
* Repetition analysis
* Redundancy analysis
* Pleonasm analysis

The information is summarized and injected into a prompt used by the language model.

The model then:

* Removes redundancy
* Simplifies wording
* Merges overlapping ideas
* Improves readability
* Preserves meaning

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
