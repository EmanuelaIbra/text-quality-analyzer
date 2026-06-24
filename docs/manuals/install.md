# Installation Guide

## Requirements

- Python 3.11+
- Ollama
- Git
- Ngrok (optional for public API access)

## Clone Repository

```bash
git clone <repository-url>

```

## Install Dependencies

```bash
pip install fastapi
pip install uvicorn 
pip install pydantic 
pip install spacy 
pip install language-tool-python 
pip install pyspellchecker 
pip install nltk 
pip install numpy 
pip install sentence-transformers 
pip install torch pip install ollama
```

## Install spaCy Italian Models

The application uses Italian language models for linguistic analysis and grammar checking.

Small model:it_core_news_sm 
LArge model:it_core_news_lg

```bash
python -m spacy download it_core_news_sm 
python -m spacy download it_core_news_lg 
```

## Install NLTK Resources

```bash
python setup_nltk.py
```
Or manually:

import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")

## Install Ollama
The Text Rewrite Module uses Llama 3.1 through Ollama.

https://ollama.com

Verify installation:

ollama --version


Pull the model:

ollama pull llama3.1

Alternative models:

ollama pull llama3
ollama pull mistral
ollama pull gemma2


## Verify Installation

Test spaCy:

python -c "import spacy; spacy.load('it_core_news_lg'); print('spaCy OK')"

Test LanguageTool:

python -c "import language_tool_python; print('LanguageTool OK')"

Test PyTorch:

python -c "import torch; print(torch.__version__)"

Test Sentence Transformers:

python -c "from sentence_transformers import SentenceTransformer; print('SentenceTransformer OK')"

Test Ollama:

ollama run llama3.1


## Run FastAPI



Start the API server:

uvicorn main:app --reload

or

python main.py

The API will be available at:

http://localhost:8000

Swagger Documentation:

http://localhost:8000/docs

ReDoc Documentation:

http://localhost:8000/redoc

## Expose the APı with NGROK
Ngrok allows external applications, mobile devices, and web clients to access your local FastAPI server through a public HTTPS URL.

Install Ngrok

Download Ngrok from:

https://ngrok.com/download

Extract the downloaded archive.

For example:

C:\ngrok\
Create a Free Ngrok Account

Create a free account:

https://dashboard.ngrok.com/signup

The free plan is sufficient for development, testing, demonstrations, and a large number of API requests.

Get Your Authentication Token

After signing in:

Open the Ngrok Dashboard.
Navigate to Your Authtoken.
Copy the generated token.

Example:

https://dashboard.ngrok.com/get-started/your-authtoken
Configure Ngrok

Open a terminal and run:

ngrok config add-authtoken YOUR_TOKEN_HERE

Example:

ngrok config add-authtoken 2AbCdEfGhIjKlMnOpQrStUvWxYz

You only need to do this once.

Start the FastAPI Server

Open a terminal in the project folder and start the API:

uvicorn main:app --reload

The API will be available locally at:

http://localhost:8000
Start Ngrok

Open a second terminal.

Navigate to the Ngrok folder:

cd C:\ngrok

Start an HTTP tunnel for the FastAPI server:

.\ngrok http 8000
Public URL

Ngrok will generate a public HTTPS URL similar to:

https://abc123.ngrok-free.app

Example output:

Forwarding https://abc123.ngrok-free.app -> http://localhost:8000

You can now access your API from anywhere using:

https://abc123.ngrok-free.app
Access API Documentation Through Ngrok

Swagger UI:

https://abc123.ngrok-free.app/docs

ReDoc:

https://abc123.ngrok-free.app/redoc
Typical Startup Sequence
Activate the Python virtual environment.
Start FastAPI:
uvicorn main:app --reload
Open a new terminal.
Navigate to the Ngrok directory:
cd C:\ngrok
Start Ngrok:
.\ngrok http 8000
Copy the generated HTTPS URL.
Use the URL to access the API from external applications.
Verify Everything Is Running

Local API:

http://localhost:8000/docs

Public API:

https://your-ngrok-url.ngrok-free.app/docs

If both URLs open successfully, the API is ready for external access.
```
