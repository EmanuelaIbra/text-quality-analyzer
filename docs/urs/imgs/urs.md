### User Requirements Specification Document

##### DIBRIS – Università di Genova. Scuola Politecnica, Software Engineering Course 80154

**VERSION : 1.0**

**Authors**
Emanuela Ibra
Yakup Gürer

---

## REVISION HISTORY

| Version | Date       | Authors                         | Notes                         |
| ------- | ---------- | ------------------------------- | ----------------------------- |
| 1.0     | 09/06/2026 | Emanuela Ibra, Yakup Gürer      | First version of the document |

---

# Table of Contents

1. [Introduction](#p1)

   1. [Document Scope](#sp1.1)
   2. [Definitions and Acronyms](#sp1.2)
   3. [References](#sp1.3)
2. [System Description](#p2)

   1. [Context and Motivation](#sp2.1)
   2. [Project Objectives](#sp2.2)
3. [Requirements](#p3)

   1. [Stakeholders](#sp3.1)
   2. [Functional Requirements](#sp3.2)
   3. [Non-Functional Requirements](#sp3.3)

---

<a name="p1"></a>

# 1. Introduction

<a name="sp1.1"></a>

## 1.1 Document Scope

This document defines the user requirements for the Italian Text Quality Analyzer project.

The system is designed to assist users in improving the quality of Italian texts by automatically detecting grammatical errors, repetitions, pleonasms, redundant expressions, and stylistic issues. The system also provides AI-assisted rewriting suggestions while preserving the original meaning of the text.

The requirements described in this document represent the expected behavior of the system from the user's perspective.

---

<a name="sp1.2"></a>

## 1.2 Definitions and Acronyms

| Acronym | Definition                        |
| ------- | --------------------------------- |
| NLP     | Natural Language Processing       |
| API     | Application Programming Interface |
| LLM     | Large Language Model              |
| DRS     | Design Requirement Specification  |
| URS     | User Requirement Specification    |
| UI      | User Interface                    |
| JSON    | JavaScript Object Notation        |
| AI      | Artificial Intelligence           |

---

<a name="sp1.3"></a>

## 1.3 References

* FastAPI Documentation
* LanguageTool Documentation
* spaCy Documentation
* NLTK Documentation
* Ollama Documentation
* Google Apps Script Documentation
* Google Docs API Documentation

---

<a name="p2"></a>

# 2. System Description

<a name="sp2.1"></a>

## 2.1 Context and Motivation

Writing high-quality texts often requires identifying grammatical mistakes, unnecessary repetitions, redundant phrases, and stylistic inconsistencies.

Many existing grammar checkers focus only on syntax correction and do not provide deeper linguistic analysis. The goal of this project is to create an integrated system capable of combining grammar correction, linguistic analysis, and AI-assisted rewriting for Italian texts.

The solution is accessible both through a REST API and directly from Google Docs.

---

<a name="sp2.2"></a>

## 2.2 Project Objectives

The objectives of the project are:

* Improve grammatical correctness of Italian texts.
* Detect lexical and semantic repetition.
* Detect and remove pleonastic expressions.
* Identify redundant sentences and repeated concepts.
* Generate improved text versions using AI.
* Preserve the original meaning of the text.
* Provide integration with Google Docs.
* Provide analysis reports explaining detected issues.

---

<a name="p3"></a>

# 3. Requirements

| Priority | Meaning            |
| -------- | ------------------ |
| M        | Mandatory          |
| D        | Desirable          |
| O        | Optional           |
| E        | Future Enhancement |

---

<a name="sp3.1"></a>

## 3.1 Stakeholders

| Stakeholder        | Description                                     |
| ------------------ | ----------------------------------------------- |
| End User           | Person using the system to improve text quality |
| Student            | Uses the system for academic writing            |
| Researcher         | Uses the system for reports and publications    |
| Developer          | Maintains and extends the software              |
| Project Supervisor | Evaluates project functionality and quality     |

---

<a name="sp3.2"></a>

## 3.2 Functional Requirements

| ID    | Description                                                                           | Priority |
| ----- | ------------------------------------------------------------------------------------- | -------- |
| FR-01 | The system shall accept Italian text as input.                                        | M        |
| FR-02 | The system shall detect grammatical errors.                                           | M        |
| FR-03 | The system shall suggest grammar corrections.                                         | M        |
| FR-04 | The system shall detect repeated words.                                               | M        |
| FR-05 | The system shall detect lemma repetitions.                                            | M        |
| FR-06 | The system shall detect synonym repetitions.                                          | D        |
| FR-07 | The system shall detect pleonastic expressions.                                       | M        |
| FR-08 | The system shall suggest replacements for detected pleonasms.                         | M        |
| FR-09 | The system shall identify semantically redundant sentences.                           | M        |
| FR-10 | The system shall generate an improved version of the text using AI.                   | M        |
| FR-11 | The system shall preserve the original meaning of the text.                           | M        |
| FR-12 | The system shall provide detailed analysis reports.                                   | M        |
| FR-13 | The system shall expose its functionality through a REST API.                         | M        |
| FR-14 | The system shall support Google Docs integration.                                     | M        |
| FR-15 | The user shall be able to accept the generated rewrite and replace the original text. | M        |
| FR-16 | The user shall be able to select different rewriting styles.                          | D        |

---

<a name="sp3.3"></a>

## 3.3 Non-Functional Requirements

| ID     | Description                                                                                   | Priority |
| ------ | --------------------------------------------------------------------------------------------- | -------- |
| NFR-01 | The system shall process texts reliably without data loss.                                    | M        |
| NFR-02 | The system shall support Italian language processing.                                         | M        |
| NFR-03 | The system shall maintain the original meaning of the text during rewriting.                  | M        |
| NFR-04 | The API shall return results in JSON format.                                                  | M        |
| NFR-05 | The Google Docs interface shall be easy to use.                                               | M        |
| NFR-06 | The system shall provide modular and maintainable code.                                       | M        |
| NFR-07 | The system shall be extensible to support additional languages.                               | E        |
| NFR-08 | The system shall support future replacement of the LLM model.                                 | E        |
| NFR-09 | The system should process medium-sized texts in less than 60 seconds under normal conditions. | D        |
| NFR-10 | The system shall run on standard personal computers without requiring cloud infrastructure.   | D        |
