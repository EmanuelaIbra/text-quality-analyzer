# Italian Pleonasm Dictionary

## Overview

The Italian Pleonasm Dictionary is a rule-based linguistic resource used by the Italian Text Quality Analyzer to identify redundant expressions in Italian texts.

A pleonasm is a linguistic construction where one or more words repeat information that is already implied by another word or expression.

Examples:

* `nuova innovazione`
* `risultato finale`
* `collaborare insieme`
* `pianificazione futura`

The goal of this dictionary is to improve text quality by detecting unnecessary redundancy and suggesting more concise alternatives.

---

# Why We Use a Dictionary

Many pleonastic expressions cannot be reliably detected using only semantic similarity or language models.

For example:

```text
nuova innovazione
```

The word *innovazione* already implies something new, therefore the adjective *nuova* is redundant.

Similarly:

```text
collaborare insieme
```

The verb *collaborare* already implies working together.

These patterns are language-specific and are best handled through a curated linguistic dictionary.

---

# Dictionary Structure

The dictionary is stored in JSON format and organized into linguistic categories.

Example:

```json
{
  "pleonasmo": "nuova innovazione",
  "forma_corretta": "innovazione",
  "spiegazione": "Innovazione implica già una novità.",
  "variante_corretta": "Innovazione."
}
```

Each entry contains:

| Field             | Description                |
| ----------------- | -------------------------- |
| pleonasmo         | Redundant expression       |
| forma_corretta    | Suggested correction       |
| spiegazione       | Linguistic explanation     |
| variante_corretta | Example of corrected usage |

---

# Categories

The dictionary currently contains multiple categories, including:

* Movement verbs with redundant adverbs
* Redundant pronouns
* Redundant adjectives and nouns
* Redundant adverbs
* Acronym redundancies
* Literary and rhetorical pleonasms
* Common spoken expressions
* Regional and dialectal expressions
* Redundant negations

This organization makes the dataset easier to maintain and extend.

---

# Dictionary Creation Process

The dictionary was created using a hybrid approach.

## Step 1 — Linguistic Sources

Initial examples were collected from:

* Accademia della Crusca
* Italian grammar references
* Italian writing style guides
* Common language usage examples

These sources provide reliable examples of pleonastic constructions.

---

## Step 2 — LLM-Assisted Generation

Large Language Models (LLMs) such as ChatGPT and Gemini were used to generate additional candidate entries.

The purpose of the LLM is not to replace linguistic validation, but to accelerate dataset expansion.

### Example Prompt 1

```text
Generate 20 common Italian pleonasms.

For each entry provide:
- pleonasmo
- forma_corretta
- spiegazione
- variante_corretta

Return only valid JSON.
```

Example generated output:

```json
{
  "pleonasmo": "nuova innovazione",
  "forma_corretta": "innovazione",
  "spiegazione": "Innovazione implica già una novità.",
  "variante_corretta": "Innovazione."
}
```

---

### Example Prompt 2

```text
Generate Italian pleonasms commonly found in business reports.

Examples:
- risultato finale
- pianificazione futura
- collaborazione insieme

Return JSON only.
```

This prompt helps generate domain-specific entries.

---

### Example Prompt 3

```text
Analyze the following Italian text and extract possible pleonasms.

For every detected pleonasm provide:
- category
- redundant expression
- suggested correction
- explanation

Return JSON only.
```

This prompt allows the discovery of new examples from real documents.

---

### Example Prompt 4

```text
Generate 100 Italian pleonasms grouped by category.

Possible categories:
- movement verbs
- redundant adjectives
- redundant adverbs
- business language
- academic writing
- common spoken expressions

Return valid JSON using the existing schema.
```

---

# Human Validation

Generated entries are not automatically accepted.

Every generated candidate should be reviewed before being added to the final dataset.

Workflow:

```text
LLM
↓
Candidate Entries
↓
Human Review
↓
Validated Entries
↓
Final Dictionary
```

This process improves quality while allowing the dictionary to scale efficiently.

---

# Scalability

The dictionary structure is intentionally designed to be scalable.

The JSON schema remains unchanged regardless of the number of entries.

New examples can be added through:

* Manual linguistic research
* Academic sources
* Writing style guides
* LLM-assisted generation
* Real-world text analysis

This allows the resource to grow from hundreds to thousands of entries while maintaining consistency.

---

# Role in the Italian Text Quality Analyzer

The Pleonasm Dictionary is only one component of the complete system.

The full analyzer consists of:

1. Grammar checking
2. Pleonasm detection
3. Repeated word detection
4. Sentence redundancy detection
5. Text rewriting and optimization

The pleonasm module uses deterministic rule-based matching through the JSON dictionary, while other modules rely on NLP techniques such as:

* Lemmatization
* Semantic similarity
* Word overlap analysis
* Redundancy detection

Combining rule-based linguistic knowledge with NLP methods provides a more robust text quality analysis system.

---

# Summary

The Italian Pleonasm Dictionary is a structured, scalable linguistic resource designed to identify redundant expressions in Italian texts.

Its development combines:

* Linguistic knowledge
* Rule-based resources
* LLM-assisted generation
* Human validation

This hybrid approach allows continuous expansion of the dictionary while maintaining linguistic quality and consistency.
