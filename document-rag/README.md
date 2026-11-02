# Document question answering

A small, local question-answering baseline over fictional employee-support documents. I built it to study passage retrieval, source attribution, and when a system should decline to answer.

## Design

The script reads the text files in `docs/`, splits them into passages, and indexes the passages with TF-IDF. For each question, cosine similarity identifies the best match. The answer is the retrieved passage, accompanied by its source filename and similarity score. A configurable threshold returns an abstention instead of a weak match.

There is no generative language model in this implementation. Calling it a full LLM-based RAG system would be misleading.

## Try it

```bash
python -m pip install -r requirements.txt
python rag.py "How do I reset my password?"
python rag.py "What is the leave carryover rule?"
python evaluate.py
python -m pytest -q
```

## Evaluation

The verified GitHub Actions run on November 1, 2026 reported **12/12 correct top-1 source-document matches** on the included synthetic evaluation questions. Eleven unit tests passed.

The evaluation is intentionally small. It measures retrieval of the expected document, not whether an answer is complete or appropriate in a real workplace.

## Next improvements

Add harder questions with overlapping topics, an out-of-scope set, and passage-level relevance labels. After that, compare TF-IDF with embedding-based retrieval before introducing answer generation.
