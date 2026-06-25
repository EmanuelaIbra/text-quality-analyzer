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

The Grammar Correction Module identifies, analyzes, and corrects grammatical issues in Italian text. It combines rule-based correction via LanguageTool with deep linguistic analysis via spaCy's dependency parser and morphology tagger to produce accurate, explainable feedback.

### Main Component

GrammarCorrector — a single class that orchestrates both engines, deduplicates their overlapping findings, and returns a unified issue list.

### Architecture Overview

Raw Italian text│├──► LanguageTool engine   →  spelling, punctuation, typography│└──► spaCy pipeline        →  16 grammar agreement rules│├── Dependency parse  (subject/verb/object relations)├── Morphology tags   (Gender, Number, VerbForm, Case…)└── Lookup tables     (exception nouns, essere-verbs…)│└── Dynamic lexicon builder (extends tables at startup)│└──► Deduplication  →  one issue per character span, highest-priority wins│└──► Structured output

### Responsibilities

Execute LanguageTool analysis on Italian textAutomatically apply LanguageTool corrections to produce a corrected stringParse and classify LanguageTool matches into spelling, punctuation, or grammarRun 16 spaCy dependency-parse rules to detect agreement and structural errorsExtend exception noun and verb lookup tables dynamically at startup using the model's own morphologyDeduplicate findings when multiple rules flag the same character spanReturn a structured payload with the original text, corrected text, and a full issue list



Inputs

Raw Italian text (string)



### Processing Flow

Validate input is not empty

Run LanguageTool → get raw matches

Auto-correct text using LanguageTool matches

Parse LanguageTool matches → normalised issue dicts

Run spaCy rules 1–16 → normalised issue dicts

Merge all issues

Deduplicate by character offset (specific rule beats generic)

Return unified result dict

### Lookup Tables

Three seed sets are defined at module level and extended once at startup by _build_dynamic_lexicons().

### _MASCULINE_NOUNS

Nouns whose surface form ends in -a but are grammatically masculine — a systematic trap for Italian learners because the -a ending normally signals feminine gender.

Why they are masculine: most come from Greek neuter nouns (ending -μα, -τα) that entered Latin as masculine and were inherited by Italian.

PatternExamplesRuleGreek -ma loansproblema, sistema, tema, clima, programma, schema, dogma, enigma, trauma, plasmaMasculine despite -a endingMedical -omamelanoma, carcinoma, glaucoma, sarcoma, adenomaGreek neuter origin-ta loansatleta, pianeta, poeta, profeta, cometaGreek -της/-τής masculine-ista professionsartista, giornalista, dentista, protagonistaCommon gender; masculine when referent is maleIndeclinable loanskoala, panda, gorilla, safari, alibiMasculine by convention

### _FEMININE_NOUNS

Nouns that look masculine (end in -o, -e, or another non--a form) but are grammatically feminine.

PatternExamplesNoteShort clippingsauto ← automobile, foto ← fotografia, moto ← motocicletta, radio, metro, biciGender inherited from the full wordIrregular historicalmanoComes from Latin manus (4th declension, feminine)-ione / -zione / -sionenazione, situazione, decisione, passione, visioneAll words in this class are feminine — very high frequency-tà / -tùcittà, libertà, verità, qualità, universitàAll words in this class are feminineGreek -sicrisi, tesi, analisi, sintesi, ipotesi, diagnosiUninflected; always feminine

### _ESSERE_VERBS

Verbs that form compound tenses (passato prossimo, trapassato, etc.) with the auxiliary essere instead of avere. The past participle of these verbs must agree in gender and number with the subject — the source of errors like siamo tornato (should be tornati).

CategoryExamplesLinguistic reasonCore motionandare, venire, partire, arrivare, tornare, uscire, entrare, salire, scendere, cadereUnergative / change-of-location verbsChange of statenascere, morire, diventare, crescere, guarire, invecchiareInchoative / resulting-state verbsCopulas & stativesessere, stare, sembrare, parere, restare, rimanereNon-agentive stativesImpersonal / weatherpiovere, nevicare, succedere, accadere, piacere, mancareNo volitional agentAppearance/disappearanceapparire, comparire, scomparire, sorgere, tramontareTelic intransitivesReflexives (all)alzarsi, lavarsi, vestirsi, svegliarsi, sedersiAll reflexive verbs take essere

### Dynamic Lexicon Builder (_build_dynamic_lexicons)

Called once during init after the spaCy model loads. Runs the model over a 60-sentence reference corpus and applies three discovery rules:

What it discoversHowNew masculine exception nounsToken ends in -a AND spaCy tags Gender=Masc → add lemma to _MASCULINE_NOUNSNew feminine exception nounsToken ends in -o or -e AND spaCy tags Gender=Fem → add lemma to FEMININE_NOUNSNew essere-verbsVERB token has a child with dep=aux and lemma=essere → add lemma to _ESSERE_VERBS

To extend coverage: add more Italian sentences to _REFERENCE_CORPUS inside the method — no code changes needed.

### spaCy Grammar Rules

All 16 rules follow the same pattern: walk the spaCy dependency tree, extract morphological features (Gender, Number, VerbForm, etc.) from relevant token pairs, compare them, and emit a standardised issue dict when they disagree.

### Rule 1 — Noun Agreement (determiner / adjective ↔ noun)

Catches: Gender or number mismatch between a pre-nominal determiner or adjective and its head noun.

Examples: qualche libri → qualche libro · bella ragazzo → bello ragazzo

Logic:

For each token with dep_="det" or dep_="amod":head = token.headif head.pos_ in (NOUN, PROPN):compare token.morph[Gender] with head.morph[Gender]compare token.morph[Number] with head.morph[Number]→ flag mismatches

### Rule 2 — Subject–Verb Agreement

Catches: Number mismatch between a nominal subject and its finite verb or auxiliary. Handles all three parse patterns spaCy produces for Italian.

Examples: Gli studenti va a scuola · Il professore hanno spiegato · Alcuni studenti era interessati

Logic:

For each token with dep_="nsubj" or dep_="nsubj":head = token.head  (may be VERB or AUX)candidates = [head]if head.pos_ == AUX:also add head.head (the main verb) and its AUX siblingsFor each candidate finite word:compare subject.morph[Number] with finite.morph[Number]→ flag mismatches (deduplicated by seen_pairs set)

Why three patterns? spaCy Italian attaches the subject to the main verb (Pattern A), to the auxiliary in compound tenses (Pattern B), or directly to an essere-auxiliary in copular constructions (Pattern C). The rule walks up one level when the direct head is an AUX to cover all cases.

### Rule 3 — Possessive–Noun Agreement

Catches: Gender or number mismatch between a possessive determiner and its head noun.

Examples: nostri progetto → nostro progetto · mia fratello → mio fratello

Logic:

For each token where dep_="det" OR (pos_=DET AND morph[Poss]=Yes):head = token.headif head.pos_ in (NOUN, PROPN):compare gender and number→ flag mismatches

### Rule 4 — Auxiliary–Past Participle Agreement

Catches: Gender or number mismatch between the subject and the past participle in essere-auxiliary compound tenses.

Italian rule: When the auxiliary is essere, the past participle must agree with the subject. This does not apply to avere constructions.

Examples: siamo tornato → siamo tornati · Maria è andato → Maria è andata

Logic:

For each VERB token with VerbForm=Part (past participle):find AUX child with lemma "essere" or "venire"if none found, check if token.head is an essere-auxiliaryfind nsubj of the predicate (on the participle or the AUX)compare subject.morph[Gender/Number] with participle.morph[Gender/Number]→ flag mismatches

### Rule 5 — Article–Noun Agreement

Catches: Gender or number mismatch between a definite or indefinite article and its head noun. Also handles exception nouns from _MASCULINE_NOUNS and _FEMININE_NOUNS.

Examples: un ragazza → una ragazza · la problema → il problema

Logic:

For each DET token with PronType=Art:head = token.headif head.pos_ in (NOUN, PROPN):if head.lemma in _MASCULINE_NOUNS and article is feminine → flagif head.lemma in _FEMININE_NOUNS and article is masculine → flagelse compare morph[Gender] and morph[Number] normally→ flag mismatches

### Rule 6 — Post-nominal Adjective Agreement

Catches: Gender or number mismatch for adjectives that appear after their noun (the common Italian position for most descriptive adjectives).

Examples: case grande → case grandi · problemi complesso → problemi complessi

Logic:

For each token with dep_="amod" and head.pos_ in (NOUN, PROPN):if token.i > head.i:  ← adjective comes AFTER the noun in textcompare token.morph[Gender/Number] with head.morph[Gender/Number]→ flag mismatches

The token.i > head.i guard distinguishes post-nominal adjectives (una casa grande) from pre-nominal ones (una grande casa) which are already handled by Rule 1.

### Rule 7 — Preposition Contraction Errors

Catches: Bare preposition + article written separately where Italian requires a contracted preposizione articolata.

Examples: di il libro → del libro · a la stazione → alla stazione · in il parco → nel parco

Logic:

Walk token list as sliding pairs (token, next_token):prep = token.text.lower()art  = next_token.text.lower()contracted = _PREP_ARTICLE_CONTRACTIONS.get((prep, art))if contracted exists → flag the preposition, suggest contracted form

The lookup table _PREP_ARTICLE_CONTRACTIONS covers all 32 combinations of the four main contracting prepositions (di, a, da, in, su) with all article forms (il, lo, la, i, gli, le, l').

### Rule 8 — Verb–Preposition Collocation Errors

Catches: Wrong preposition used after a verb that requires a specific one by Italian convention.

Examples: vado in casa → vado a casa · parlare su qualcosa → parlare di qualcosa

Logic:

For each VERB/AUX token whose lemma is in VERB_PREP_RULES:expected_prep = rule["expected"]wrong_preps   = rule["wrong"]for each oblique child (dep in obl, obl, nmod, advmod):for each "case" grandchild with pos_=ADP:if grandchild.text.lower() in wrong_preps → flag

Current collocation rules: andare→a, pensare→a, ringraziare→per, dipendere→da, parlare→di.

### Rule 9 — Predicate Adjective Agreement

Catches: Gender or number mismatch between the subject and a predicate adjective linked by a copular verb (essere, sembrare, diventare, restare, …).

Examples: Lei era simpatico → simpatica · La professoressa sembra stanco → stanca

Logic — two parse patterns handled:

spaCy Italian produces two different tree structures for this construction:

Pattern A (verb-headed):era(AUX, ROOT) ←── simpatico(ADJ, dep="attr")era ←── Lei(PRON, dep="nsubj")

Detection: ADJ.head.lemma in COPULAR_VERBSsubject = child of head with dep="nsubj"

Pattern B (adjective-headed / copula-as-child):simpatico(ADJ, ROOT) ←── era(AUX, dep="cop")simpatico ←── Lei(PRON, dep="nsubj")

Detection: ADJ has a child with dep_="cop" and lemma in COPULAR_VERBSsubject = child of ADJ with dep="nsubj"

Both patterns are checked for every ADJ token; gender and number are compared once a subject is found.

### Rule 10 — Clitic Pronoun Elision

Catches: The clitic pronoun lo or la used before a vowel-initial verb, where Italian requires elision to l'.

Examples: Lo ho visto → L'ho visto · La avevo incontrata → L'avevo incontrata

Logic:

Walk token list as sliding pairs:if token.text.lower() in ("lo", "la"):next_token = tokens[i + 1]if next_token.pos_ in (VERB, AUX) and next_token.text[0] is a vowel:→ flag, suggest "l'" + next_token.text

### Rule 11 — Partitive Article Agreement

Catches: Gender or number mismatch between a partitive article (del, della, dei, delle, degli) and its head noun.

Examples: del mele → delle mele · delle pane → del pane

Logic:

_PARTITIVE_MORPH = {"del":  (Masc, Sing), "dello": (Masc, Sing),"della":(Fem,  Sing), "dei":   (Masc, Plur),"degli":(Masc, Plur), "delle": (Fem,  Plur),"dell'":(None, Sing),  # gender ambiguous}For each DET token whose text is in _PARTITIVE_MORPH:derive (art_gender, art_number) from the mapcompare with head.morph[Gender/Number]→ flag mismatches, suggest correct partitive form

### Rule 12 — Missing Reflexive Clitic

Catches: Verbs that are inherently reflexive used without their required clitic pronoun (mi, ti, si, ci, vi).

Examples: Mario lava ogni mattina → Mario si lava ogni mattina

Logic:

For each VERB/AUX token whose lemma is in REFLEXIVE_LEMMA_MAP:if any child has dep="obj" or dep_="iobj" → suppress (transitive use)if any child has pos_=PRON and Clitic=Yes → suppress (clitic present)if any token in 2-token left window has Clitic=Yes → suppressotherwise → flag, suggest reflexive form

The suppression guards are important: lava i piatti (washes the dishes) is a valid transitive use of lavare and should not be flagged.

### Rule 13 — Double Negation (Missing non)

Catches: Sentences containing a negative polarity item (niente, nessuno, mai, nemmeno, né, …) without the required non before the finite verb.

Italian rule: Italian uses concordance negation — both non before the verb AND the NPI are required. Unlike English, omitting non is ungrammatical.

Examples: Ho visto niente → Non ho visto niente · Viene mai → Non viene mai

Logic:

For each sentence:if sentence contains any word in _NEGATIVE_POLARITY_WORDS:for each finite VERB/AUX:check for "non" as advmod child OR in 3-token left windowif no negation marker found → flag the verb(one flag per sentence to avoid noise)

### Rule 14 — Interrogative Word Order

Catches: Subject pronoun appearing between a WH-word and the verb in a direct question — ungrammatical in Italian (unlike English).

Examples: Cosa tu fai? → Cosa fai? or Cosa fai tu?

Logic:

For each sentence ending in "?":for each WH-word token (cosa, chi, come, dove, quando, perché, …):scan next 3 tokens for a subject pronoun (io, tu, lui, lei, noi, voi, loro)if pronoun.dep_ in (nsubj, nsubj) AND a verb follows it:→ flag the pronoun

### Rule 15 — Gerund Subject Mismatch

Catches: Adjective inside a gerund clause that disagrees in gender or number with the subject of the main clause. In Italian, the implied subject of a gerund must always match the main clause subject.

Examples: Essendo stanca, lui uscì → Essendo stanco, lui uscì

Logic:

Find the ROOT verb of the sentence → find its nsubj → get subject gender/numberFor each VERB token with VerbForm=Ger (gerund):for each ADJ child of the gerund:compare ADJ.morph[Gender/Number] with main subject gender/number→ flag mismatches

### Rule 16 — Modal + Non-Infinitive

Catches: A verb following a modal that is not in the infinitive form — learners often use a past participle or conjugated form instead.

Examples: Voglio andati → Voglio andare · Puoi venuto → Puoi venire

Logic:

For each VERB/AUX token whose lemma is in MODAL_VERBS:for each child with dep in (xcomp, ccomp, obj):if child.pos_ in (VERB, AUX):verb_form = child.morph[VerbForm]if "Inf" not in verb_form → flag, suggest child.lemma_ (infinitive)

Modal verbs covered: volere, potere, dovere, sapere, riuscire, osare, desiderare, preferire, sperare, tentare, cercare.

### Deduplication

Multiple rules can fire on the same token — for example, Rule 1 (generic noun agreement) and Rule 5 (article–noun agreement) both flag the same article. The correct_text method deduplicates by character offset before returning results.

Priority order (higher wins per span):

PrioritySource0 — lowestLanguageTool (generic)1spaCy generic rules (Rule 1, Rule 6)2 — highestspaCy specific rules (Rules 3–5, 7–16)

A specific rule message like "Article–noun gender mismatch: 'La' does not agree with 'problema'" replaces the generic "'La' may not agree with 'problema'" for the same span.

### Outputs

python{"original":  str,   # unchanged input"corrected": str,   # LanguageTool auto-corrected version"polished":  str,   # same as corrected (hook for future post-processing)"matches": [        # deduplicated, ordered list of issues{"source":     "LanguageTool" | "spaCy","issue_type": "grammar" | "spelling" | "punctuation","rule":       str,        # e.g. "SPACY_ARTICLE_NOUN_AGREEMENT""message":    str,        # human-readable explanation"offset":     int,        # character position in original text"length":     int,        # length of the flagged span"wrong_text": str,        # the flagged token(s)"context":    str,        # surrounding text window"suggestions": [str],     # replacement candidates (may be empty)},…]}
### External Dependencies

* language_tool_python 
* Italian spaCy model (it_core_news_lg)

---

# 5 Text Analysis Module

The Text Analysis Module evaluates the quality of corrected Italian text by identifying repetition, pleonasms, semantic similarity, and redundant sentence structures.

It combines rule-based text normalization, spaCy linguistic analysis, Italian WordNet synonym lookup, static pleonasm dictionaries, and multilingual sentence embeddings.

---

## Components

| Component | Class / Function | Role |
|---|---|---|
| Repetition analyzer | `RepetitionAnalyzer` | Lexical, lemma-based, synonym-based, and statistical repetition analysis |
| Redundancy checker | `TextRedundancyChecker` / `analyze_text()` | Pleonasm detection, similar-word detection, redundant sentence detection |
| Pleonasm helpers | `load_pleonasm_entries()`, `find_pleonasms()`, `apply_pleonasm_replacements()` | Load JSON dictionary, match lemma patterns, apply corrections |
| Redundancy helpers | `find_redundant_sentences()`, `find_similar_words()`, `find_repeated_words()` | Detect duplicate and near-duplicate sentences and word pairs |
| Text normalization | `normalize_spacing()`, `normalize_for_dedup()`, `split_sentences()` | Prepare text for consistent comparison |
| Sentence encoding | `get_sentence_model()`, `encode_sentences()` | Load and run the multilingual transformer model |
| Entry point | `analyze_text()` | Run all checks and return one structured report |
| CLI helper | `print_report()` | Pretty-print results for manual debugging |

---

## Pipeline Overview

```
Raw corrected Italian text
        │
        ▼
  normalize_spacing()
        │
        ▼
  spaCy parse (it_core_news_lg)
        │
        ├──────────────────────────────────────────────────────────┐
        │                                                          │
        ▼                                                          ▼
RepetitionAnalyzer.analyze()                              analyze_text()
  ├─ remove_direct_word_repetition()                        ├─ find_pleonasms()
  ├─ remove_repeated_sentences()                            │    └─ lemma pattern sliding window
  ├─ repeated_words()                                       ├─ find_repeated_words()
  ├─ top_repeated_words()                                   ├─ find_similar_words()
  ├─ lexical_diversity_score()                              │    └─ spaCy vector cosine similarity
  ├─ repetition_ratio()                                     └─ find_redundant_sentences()
  ├─ highlight_repetition()                                      ├─ exact duplicate check
  ├─ lemma_repetition()                                          ├─ sentence-transformer encoding
  └─ synonym_repetition()                                        └─ lexical redundancy fallback
       └─ Italian WordNet lookup
        │                                                          │
        └──────────────────────────┬───────────────────────────────┘
                                   ▼
                     Structured quality report dict
```

---

## Shared Configuration

### `SEMANTIC_POS`

Defines the token types considered meaningful for repetition and semantic checks.

```python
SEMANTIC_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
```

Only nouns, verbs, adjectives, adverbs, and proper nouns are treated as semantic content.

---

### `REDUNDANCY_THRESHOLDS`

Defines similarity thresholds used to classify redundant sentence pairs.

```python
REDUNDANCY_THRESHOLDS = {
    "duplicate":       0.95,
    "manual_review":   0.85,
    "merge_candidate": 0.70,
}
```

| Score range | Label | Meaning |
|---|---|---|
| `>= 0.95` | `duplicate` | Sentences are effectively identical |
| `>= 0.85` | `manual_review` | Highly similar; should be reviewed manually |
| `>= 0.70` | `merge_candidate` | Related enough to consider merging |
| `< 0.70` | `related` | Related but below all redundancy thresholds |

---

### Sentence Model

```python
SENTENCE_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
```

Used to compare sentence meanings through normalized vector embeddings. Runs on CPU; loaded once via `@lru_cache`.

---

## External Dependencies

| Library | Purpose |
|---|---|
| `re` | Tokenization, spacing normalization, duplicate-word removal, sentence splitting |
| `spacy` | Italian tokenization, POS tagging, lemmatization, sentence segmentation, semantic vectors |
| `nltk.wordnet` | Italian synonym detection |
| `sentence_transformers` | Multilingual sentence similarity detection |
| `numpy` | Vector normalization and cosine similarity |
| `json` | Loading static pleonasm dictionaries |
| `collections.Counter` | Counting repeated words |
| `collections.defaultdict` | Grouping words by lemma |
| `functools.lru_cache` | Caching WordNet synonyms and the sentence-transformer model |
| `pathlib.Path` | Resolving the pleonasm JSON file path |

---

## WordNet Synonym Setup

### `_ensure_wordnet()`

Downloads the required NLTK WordNet packages on first use only.

Packages: `wordnet`, `omw-1.4`.

```
if WordNet is already ready → return

for each required package:
    try to find it locally
    if missing → download silently

mark WordNet as ready
```

---

### `get_italian_synonyms(word)`

Returns cached Italian WordNet synonyms for a lowercase word.

```
ensure WordNet is available

for each Italian synset of the word:
    for each Italian lemma in the synset:
        replace underscores with spaces
        lowercase the synonym
        add to synonym set

return synonyms
```

Caching: `@lru_cache(maxsize=4096)` — avoids repeated WordNet lookups for the same word.

---

## RepetitionAnalyzer

Performs lexical, lemma-based, synonym-based, and statistical repetition analysis on Italian text.

### Initialization

```python
def __init__(self, nlp=None):
    self.nlp        = nlp or spacy.load("it_core_news_lg")
    self.stop_words = self.nlp.Defaults.stop_words
    self._SYNONYM_SKIP = {"ieri", "sera", "molto", "anche", "però", "ormai"}
```

`_SYNONYM_SKIP` prevents noisy synonym detections for frequent or context-dependent words.

---

### Text Utilities

#### `tokenize(text)`

Splits text into lowercase word tokens using the pattern `r"\b[a-zA-ZÀ-ÿ]+\b"`. Handles standard alphabetic words and Italian accented characters.

```
"La città è bella." → ["la", "città", "è", "bella"]
```

#### `get_content_words(text)`

Filters tokens to keep only meaningful words: not a stop word and length greater than 2. Removes articles, prepositions, conjunctions, and short function words.

---

### Cleaning Pipeline

#### `remove_direct_word_repetition(text)`

Removes immediately repeated words using the regex `r"\b(\w+)(\s+\1\b)+"` (case-insensitive).

```
casa casa          → casa
molto molto bello  → molto bello
```

#### `remove_repeated_sentences(text)`

Removes exact duplicate sentences while preserving original order.

```
Oggi piove. Oggi piove. Domani esco.
→ Oggi piove. Domani esco.
```

#### `clean(text)`

Runs both cleaning steps in sequence: direct word repetition removal → duplicate sentence removal.

---

### Repetition Statistics

#### `repeated_words(text)` → `dict`

Returns content words appearing more than once.

```python
{"casa": 3, "bello": 2}
```

#### `top_repeated_words(text, limit=10)` → `list`

Returns the most frequent repeated content words, sorted by frequency descending.

```python
[("casa", 3), ("bello", 2)]
```

#### `lexical_diversity_score(text)` → `float`

Type-Token Ratio of content words as a percentage.

```
unique_content_words / total_content_words × 100
```

Higher score = richer vocabulary. Lower score = more repetition. Returns `100.0` if no content words found.

#### `repetition_ratio(text)` → `float`

Percentage of redundant extra occurrences among content words.

```
sum(count − 1 for each repeated word) / total_content_words × 100
```

For `casa` appearing 3 times, only 2 occurrences count as redundant.

#### `highlight_repetition(text)` → `str`

Wraps repeated content words in `<lexrep>` tags.

```xml
La <lexrep>casa</lexrep> è bella. La <lexrep>casa</lexrep> è grande.
```

---

### Linguistic Repetition

#### `lemma_repetition(text)` → `dict`

Groups repeated words by their spaCy lemma. Detects grammatical variations of the same root.

```
gatto, gatti     → lemma: gatto
andare, andiamo  → lemma: andare
```

Only tokens with POS in `SEMANTIC_POS`, not stop words, and not punctuation are considered.

```python
{"gatto": ["gatto", "gatti"]}
```

#### `synonym_repetition(text, window=None)` → `list`

Detects synonym-like repetition using Italian WordNet. By default, compares words only within the same sentence. Words of length ≤ 3 and words in `_SYNONYM_SKIP` are skipped.

```python
[{"pair": ("nuovo", "recente"), "sentence": "È un sistema nuovo e recente."}]
```

If `window` is provided, the analyzer also checks nearby words across sentence boundaries within the next `window` tokens. Cross-sentence matches are labeled accordingly.

---

### Main Repetition Analysis

#### `analyze(text)` → `dict`

Runs all repetition checks and returns one dictionary.

```
1. clean text
2. detect lemma repetition
3. detect synonym repetition
4. calculate repeated words
5. calculate top repeated words
6. calculate lexical diversity score
7. calculate repetition ratio
8. highlight repeated words
```

Output keys:

| Key | Type | Description |
|---|---|---|
| `cleaned_text` | `str` | Text after direct-repetition and duplicate-sentence removal |
| `had_direct_repetition` | `bool` | Whether any direct word repetition was found and removed |
| `repeated_words` | `dict` | Content words appearing more than once with counts |
| `top_repeated_words` | `list` | Most frequent repeated words, sorted descending |
| `lexical_diversity_score` | `float` | Type-Token Ratio as a percentage |
| `repetition_ratio` | `float` | Percentage of redundant extra word occurrences |
| `highlighted_repetition` | `str` | Text with `<lexrep>` tags around repeated words |
| `lemma_repetition` | `dict` | Repeated words grouped by lemma |
| `synonym_repetition` | `list` | Synonym pairs found within sentences |
| `has_synonym_repetition` | `bool` | Whether any synonym repetition was detected |

---

## TextRedundancyChecker / `analyze_text`

Performs pleonasm detection, similar-word detection, and redundant sentence detection.

### Text Normalization

#### `normalize_spacing(text)`

Normalizes whitespace and punctuation spacing before analysis.

```
1. collapse repeated whitespace into one space
2. add missing space after ".", "!", "?" when followed by uppercase
3. remove unnecessary spaces before punctuation
4. trim leading and trailing spaces
```

```
Ciao!Come stai ?  Bene.  →  Ciao! Come stai? Bene.
```

#### `normalize_for_dedup(sentence)`

Produces a simplified form for exact duplicate detection: lowercase → trim → remove punctuation → preserve accented characters → collapse spaces.

```
"Oggi piove!"  →  "oggi piove"
```

#### `split_sentences(text)`

Normalizes spacing, then splits on `.`, `!`, `?` boundaries, discarding empty strings.

---

### Pleonasm Detection

#### `load_pleonasm_entries(json_path=None)` → `list`

Loads Italian pleonasm examples from the JSON data file at `DATA_DIR / "italian_pleonasms.json"`.

Expected JSON structure:

```python
{
  "categories": {
    "category_name": {
      "examples": [
        {
          "pleonasmo":        "...",
          "forma_corretta":   "...",
          "spiegazione":      "...",
          "variante_corretta":"..."
        }
      ]
    }
  }
}
```

Returns a flat list of entries with keys: `phrase`, `replacement`, `category`, `explanation`, `correct_variant`. Empty phrases are skipped. If the file does not exist, a warning is printed and an empty list is returned.

#### `_build_lemma_patterns(entries, nlp)` → `list`

Converts pleonasm phrases into spaCy lemma sequences so that inflected variants are also matched.

```
"risultato finale"  →  lemmas: ["risultato", "finale"]
"risultati finali"  →  same match
```

Punctuation, spaces, and empty tokens are ignored during lemma extraction.

#### `get_lemma_patterns(nlp)` → `list`

Returns cached lemma patterns for the active spaCy model. Cache key is `(model_name, model_version)`. Patterns are built once on first call and reused.

#### `warmup_pleonasm_cache(nlp)`

Prebuilds pleonasm patterns at application startup by calling `get_lemma_patterns(nlp)`.

#### `find_pleonasms(text, nlp)` → `list`

Finds pleonastic expressions by sliding a window of lemmas over the parsed text.

```
normalize spacing → parse with spaCy → strip punctuation/space tokens
→ extract lemmas
→ for each pattern: slide window → if lemmas match → record surface text
```

Duplicate matches (same phrase found twice) are deduplicated by a uniqueness key.

Output example:

```python
{
  "phrase":        "risultati finali",
  "base_phrase":   "risultato finale",
  "replacement":   "risultato",
  "category":      "ridondanza",
  "explanation":   "...",
  "correct_variant": "..."
}
```

#### `apply_pleonasm_replacements(text, pleonasms)` → `str`

Replaces each detected pleonasm with its corrected form using a case-insensitive word-boundary regex. If a replacement contains `/`, only the first option is used. Spacing is normalized before and after substitution.

---

### Repeated Words Inside Sentences

#### `find_repeated_words(text, nlp, doc=None)` → `list`

Detects repeated semantic lemmas within individual sentences. Uses `doc.sents` when spaCy sentence boundaries are available; falls back to manual splitting otherwise.

For each sentence, content tokens (POS in `SEMANTIC_POS`, not stop word, not punctuation, length > 2) are extracted and their lemmas counted. Sentences with any lemma appearing more than once are returned.

```python
[{"sentence": "Il gatto e i gatti dormono.", "words": ["gatto"]}]
```

---

### Similar Word Detection

#### `find_similar_words(doc, threshold=0.75, max_tokens=80)` → `list`

Finds semantically similar word pairs using spaCy lemma vectors.

```
extract semantic content tokens → limit to max_tokens
→ get vector for each token.lemma_ → keep only tokens with valid vectors
→ normalize vectors → compute cosine similarity matrix
→ keep pairs where: threshold ≤ similarity < 1.0 and lemmas differ
→ sort by similarity descending
```

```python
[("nuovo", "moderno", 0.86), ("rapido", "veloce", 0.82)]
```

---

### Redundant Sentence Detection

#### `_sentence_lemmas(sentence, nlp=None)` → `set`

Returns content lemmas for lexical redundancy scoring. With spaCy: POS-filtered, stop-word-excluded lemmas. Without spaCy: lowercase alphabetic tokens longer than 3 characters.

#### `_lexical_redundancy_score(first, second)` → `float`

Backup score for near-duplicate sentences when the transformer score is too conservative, sentences are short, or the transformer is unavailable.

| Condition | Result |
|---|---|
| Either sentence has fewer than 3 content lemmas | `0.0` |
| Containment `≥ 0.80` and Jaccard `≥ 0.50` | `min(0.87, 0.74 + 0.16 × jaccard)` |
| Containment `≥ 0.65` and Jaccard `≥ 0.45` | `min(0.78, 0.68 + 0.14 × jaccard)` |
| Otherwise | Jaccard score |

Where containment = `overlap / smaller set size` and Jaccard = `overlap / union set size`.

#### `find_redundant_sentences(sentences, threshold=0.80, nlp=None)` → `list`

Finds redundant or mergeable sentence pairs using three signals in combination.

```
if fewer than 2 sentences → return []

normalize sentences for exact comparison
build lemma sets for lexical scoring

for each pair:
    if exact duplicate → score 1.0, label "duplicate"
    else → queue for transformer scoring

try:
    encode all sentences with sentence-transformer
    compute cosine similarity matrix
except:
    use lexical fallback only

for each non-exact pair:
    final_score = max(transformer_score, lexical_score)
    if final_score >= threshold → add pair with label

return results sorted by score descending
```

---

### Sentence Encoding

#### `get_sentence_model()`

Loads the multilingual sentence-transformer model once. Cached via `@lru_cache(maxsize=1)`. Runs on CPU.

#### `encode_sentences(sentences)` → `ndarray`

Encodes a list of sentences into normalized embedding vectors. Uses batch size 64 with progress bar disabled.

---

## Main Entry Point — `analyze_text()`

Runs all text-quality checks and returns one structured report.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | — | Corrected Italian text to analyze |
| `word_sim_threshold` | `float` | `0.82` | Minimum cosine similarity for similar-word pairs |
| `sent_sim_threshold` | `float` | `0.72` | Minimum score for redundant sentence detection |
| `max_similar_tokens` | `int` | `80` | Maximum tokens passed to `find_similar_words()` |
| `nlp` | spaCy model | `None` | If not provided, `it_core_news_lg` is loaded automatically |

**Flow:**

```
if text is empty → return empty report

if nlp not provided → load it_core_news_lg

normalize spacing
parse with spaCy
extract sentences from doc.sents
fallback to manual split if needed

run find_pleonasms()
run find_repeated_words()
run find_similar_words()
run find_redundant_sentences()
```

**Output:**

| Key | Type | Description |
|---|---|---|
| `pleonasms` | `list` | Detected pleonastic expressions with replacements |
| `repeated_words` | `list` | Sentences containing repeated lemmas |
| `similar_words` | `list` | Semantically similar word pairs with scores |
| `redundant_sentences` | `list` | Redundant sentence pairs with scores and labels |

---

## CLI Report Helper — `print_report(report)`

Pretty-prints the result of `analyze_text()` for manual debugging. Prints four sections in Italian: `PLEONASMI`, `PAROLE RIPETUTE NELLA STESSA FRASE`, `PAROLE SIMILI / QUASI SINONIMI`, `FRASI RIDONDANTI`. Each section displays findings or a message indicating no issues were found.

---

## Data Flow Between Modules

```
Raw corrected Italian text
        │
        ▼
RepetitionAnalyzer.analyze()  ──►  repetition_analysis  ──►  TextRewriter.rewrite()
        │
        ▼
analyze_text()                ──►  redundancy_report    ──►  TextRewriter.rewrite()
```

The outputs of both `RepetitionAnalyzer.analyze()` and `analyze_text()` are consumed directly by the Text Rewrite Module.

---

# 6 Text Rewrite Module

The Text Rewrite Module rewrites corrected and analyzed Italian text to improve clarity, fluency, conciseness, and readability. It coordinates pre-processing, structured prompt construction, LLM-based rewriting, and output cleanup into a single pipeline.

---

## Components

| Component | Class / Function | Role |
|---|---|---|
| Sentence pre-merger | `PreMerger` | Analyzes and prepares redundant sentence pairs before rewriting |
| Text rewriter | `TextRewriter` | Orchestrates the full rewrite pipeline |
| Prompt builder | `build_prompt()` | Builds the structured LLM prompt from all analysis results |
| Output cleaner | `clean_llm_output()` | Strips LLM preambles, notes, and bullet lists from the response |
| Sentence splitter | `split_sentences()` | Splits text into sentences on `.`, `!`, `?` boundaries |
| Text normalizer | `_normalize()` | Collapses whitespace for consistent string comparison |

---

## Pipeline Overview

```
Input text + analysis results
        │
        ▼
  PreMerger.merge()
  ├─ Normalize text and split into sentences
  ├─ Classify sentence pairs by similarity score
  │   ├─ score ≥ user_choice_threshold  →  user_choice_candidates
  │   └─ merge_threshold ≤ score < user_choice_threshold  →  merge_candidates
  ├─ Identify resolved repeated words
  └─ Return pre-merged text and metadata
        │
        ▼
  build_prompt()
  └─ Combine pre-merged text + all analysis metadata into a structured prompt
        │
        ▼
  ollama.chat()  (temperature = 0.1)
        │
        ▼
  clean_llm_output()
        │
        ▼
  Final rewritten text
```

---

## Responsibilities

### Sentence Pre-Merging — `PreMerger`

Prepares redundant sentence pairs before the text is sent to the LLM. No sentences are deleted automatically at this stage.

**Initialization parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nlp` | spaCy model | `None` | Optional; enables lemma-based content word extraction |
| `user_choice_threshold` | `float` | `0.85` | Minimum similarity to flag a pair as requiring user selection |
| `merge_threshold` | `float` | `0.65` | Minimum similarity to flag a pair as a merge candidate |

**Similarity classification:**

| Score range | Category | Action |
|---|---|---|
| `>= user_choice_threshold` | User choice candidate | Marked for user decision; never auto-deleted |
| `>= merge_threshold` and `< user_choice_threshold` | Merge candidate | Suggested for merging if it improves clarity |
| `< merge_threshold` | No action | Left unchanged |

**`merge()` return value** — a 5-tuple:

```python
(
    merged_text,            # Normalized text with no auto-deletions
    resolved_repeats,       # Repeated words no longer repeated after preprocessing
    merge_candidates,       # Pairs eligible for merging
    deleted_pairs,          # Always empty — no auto-deletion occurs
    user_choice_candidates, # Highly similar pairs awaiting user decision
)
```

**Content word extraction — `_content_words()`:**

- With spaCy: extracts lemmas of `NOUN`, `VERB`, `ADJ`, `ADV`, `PROPN` tokens, excluding stop words and punctuation.
- Without spaCy: falls back to regex tokenization with stop-word removal.

**Resolved repeat detection — `_find_resolved_repeats()`:**

After preprocessing, identifies which previously repeated words now appear only once or not at all in the remaining text. These are excluded from the rewrite prompt so the LLM is not asked to fix issues already resolved.

---

### User Choice Handling

When two sentences exceed `user_choice_threshold` (default `0.85`), the module does **not** automatically remove either one.

Each pair is added to `user_choice_candidates` with metadata:

```python
{
    "sentence_1": "...",
    "sentence_2": "...",
    "similarity": 0.91,
    "action": "scelta_utente: scegliere la frase A, la frase B oppure mantenere entrambe"
}
```

The LLM is explicitly instructed to keep both sentences unless the user has made a selection, or to merge them without losing information.

---

### Merge Candidate Detection

Sentence pairs with similarity between `merge_threshold` and `user_choice_threshold` are flagged as merge candidates.

Each candidate is structured as:

```python
{
    "sentence_1": "...",
    "sentence_2": "...",
    "similarity": 0.72,
    "action": "unire solo se migliora la chiarezza; non eliminare informazioni utili"
}
```

The LLM is instructed to merge these pairs **only** when merging improves clarity and does not drop useful information.

Pairs are sorted by descending similarity before classification. If a sentence has already been dropped from the active sentence set, it is skipped to prevent duplicate processing.

---

### Prompt Generation — `build_prompt()`

Builds a structured Italian-language prompt that gives the LLM full context for the rewrite.

**Supported rewrite modes:**

| Mode | Instruction |
|---|---|
| `concise` | Shorten and clarify without removing important information |
| `academic` | Formal, clear style; avoids unnecessarily complex vocabulary |
| `fluent` | Natural, readable prose with simple vocabulary |
| `standard` | Clear, direct, professional Italian (default fallback) |

**Prompt sections:**

| Section | Content |
|---|---|
| 1. Repeated words | Words still repeated after pre-processing, with occurrence count |
| 2. Pleonasms | Detected pleonasms and their suggested replacements |
| 3. Similar words | Word pairs with similarity `0.75 ≤ score < 1.00`; capped at ~360 characters to limit prompt length |
| 4. Auto-deleted pairs | Always empty in the current implementation |
| 5. Merge candidates | Sentence pairs eligible for merging with similarity score and action label |
| 6. User choice candidates | Highly similar pairs to be preserved or merged without auto-deletion |

**Prompt rules injected into every request:**

- Use the analysis above to guide the rewrite.
- Do not automatically delete user-choice sentences.
- Merge related sentences only when it improves clarity.
- Remove repeated words and pleonasms.
- Use simple, natural Italian — avoid rare synonyms used only for variation.
- Preserve all important information; do not add new facts.
- Return only the final rewritten text.
- Do not modify, delete, or move placeholders in the format `[[KEEP_SENTENCE_X]]`.

---

### LLM-Based Rewriting — `TextRewriter`

Orchestrates the full pipeline and calls the Ollama API.

**Initialization parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"llama3.1"` | Ollama model used for generation |
| `nlp` | spaCy model | `None` | Passed through to `PreMerger` |
| `user_choice_threshold` | `float` | `0.85` | Passed through to `PreMerger` |
| `merge_threshold` | `float` | `0.65` | Passed through to `PreMerger` |

**`rewrite()` parameters:**

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Original text to rewrite |
| `repetition_analysis` | `dict` | Output from the repetition analyzer; expects `repeated_words` key |
| `redundancy_report` | `dict` | Output from the redundancy detector; expects `redundant_sentences`, `pleonasms`, `similar_words` keys |
| `mode` | `str` | Rewrite style: `concise`, `fluent`, `academic`, or `standard` |

**LLM call settings:**

- System prompt: instructs the model to act as a professional Italian text reviewer using clear, natural, simple language.
- Temperature: `0.1` (low, for deterministic and conservative rewrites).

---

### Output Cleaning — `clean_llm_output()`

Strips unwanted content from the raw LLM response before returning the final text.

**Removed preambles** (case-insensitive, at the start of the response):

| Pattern | Example |
|---|---|
| `ecco(?: il)? testo riscritto[:\-]?` | *Ecco il testo riscritto:* |
| `testo riscritto[:\-]?` | *Testo riscritto:* |
| `versione corretta[:\-]?` | *Versione corretta:* |
| `versione migliorata[:\-]?` | *Versione migliorata:* |
| `ho riscritto il testo[:\-]?` | *Ho riscritto il testo:* |
| `certo[,!]? ecco…:` | *Certo! Ecco il testo rivisto:* |

**Removed postamble sections** (entire remainder discarded once matched):

| Marker | Examples |
|---|---|
| `modifiche` | *Modifiche apportate:* |
| `spiegazione` | *Spiegazione:* |
| `nota` | *Nota:* |
| `ho rimosso` | *Ho rimosso le ripetizioni…* |
| `ho unito` | *Ho unito le frasi…* |
| `ho corretto` | *Ho corretto…* |
| Lines starting with `* ` or `- ` | Bullet-point summaries |

**Additional cleanup:**

- Inline bullet lines (`* ...` or `- ...`) anywhere in the text are removed.
- Three or more consecutive blank lines are collapsed to two.

---

## Data Flow Between Modules

```
RepetitionAnalyzer  ──► repetition_analysis ──► TextRewriter.rewrite()
RedundancyDetector  ──► redundancy_report   ──► TextRewriter.rewrite()
                                                     │
                                               PreMerger.merge()
                                                     │
                                               build_prompt()
                                                     │
                                               ollama.chat()
                                                     │
                                           clean_llm_output()
                                                     │
                                           Final rewritten text
```


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
