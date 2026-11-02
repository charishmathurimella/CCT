# Grounded Document Question Answering

A local, reproducible retrieval-augmented question-answering baseline using synthetic employee-support documents. It does not require API keys, external model calls, or proprietary information.

## Architecture

1. Load the source documents in `docs/`.
2. Split each document into short passages.
3. Index passages with TF-IDF vectors.
4. Retrieve the most relevant passages for a question using cosine similarity.
5. Return an extractive answer with the document name and a relevance score.
6. Abstain when retrieval confidence is below a configured threshold.

This is a **retrieval and extractive answering baseline**, not a generative LLM implementation. It deliberately avoids inventing answers not supported by the documents.

## Run

```bash
pip install -r requirements.txt
python rag.py "How do I reset my password?"
python rag.py "What is the leave carryover rule?"
pytest -q
```

## Evaluation

`python evaluate.py` measures retrieval top-1 accuracy against the labeled questions in `data/evaluation.csv`. It also reports coverage and correct answers on answered queries. The evaluation examples are synthetic and small; results should not be generalized to real workplace use.

## Next engineering milestones

Add embedding-based retrieval, a held-out evaluation set, a citation-verifying answer generator, and an explicit out-of-scope benchmark. All examples are original synthetic content.
