# BioMed IR Search Engine

University project: **Information Retrieval & Data Mining** (אחזור וכריית מידע).  
Authors: Maria Alexeenko, Ilay Sabach, Victoria Golovitsky.

## Contents

- **EX_01** – NLP text analysis (tokenization, lemmatization, WordNet)
- **EX_02** – Vector space models (Binary, Frequency, TF-IDF)
- **EX_03** – Search engine UI and evaluation metrics

## Run the search UI locally

1. Generate data: `python generate_ui_data.py`
2. Serve: `python -m http.server 8000`
3. Open: http://localhost:8000 (use `index.html`)

## Data

- `all_docs_8.jsonl` – documents  
- `queries_8.jsonl` – queries  
- `qrel_8.jsonl` – relevance judgments  
