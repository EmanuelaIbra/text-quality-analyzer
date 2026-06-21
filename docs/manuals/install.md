# Installation Guide

## Requirements

- Python 3.11+
- Ollama
- Git

## Clone Repository

```bash
git clone <repository-url>
cd project
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Install spaCy Italian Models

```bash
python -m spacy download it_core_news_sm
python -m spacy download it_core_news_lg
```

## Install NLTK Resources

```bash
python setup_nltk.py
```

## Install Ollama

https://ollama.com

Pull model:

```bash
ollama pull llama3.1
```