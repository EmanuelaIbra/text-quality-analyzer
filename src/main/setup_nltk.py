# setup_nltk.py

import nltk

resources = [
    "punkt",
    "stopwords",
    "wordnet",
    "omw-1.4"
]

for resource in resources:
    nltk.download(resource)

print("NLTK setup complete.")