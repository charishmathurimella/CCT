"""Local extractive document QA with retrieval scores and source citations."""
import argparse
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS = Path(__file__).resolve().parent / "docs"


def load_documents(folder=DOCS):
    documents = []
    for path in sorted(Path(folder).glob("*.txt")):
        content = path.read_text(encoding="utf-8")
        for passage in content.split("\n\n"):
            passage = passage.strip()
            if passage and not passage.startswith("#"):
                documents.append({"source": path.name, "text": passage})
    if not documents:
        raise ValueError("No documents found")
    return documents


class Retriever:
    def __init__(self, documents=None):
        self.documents = documents if documents is not None else load_documents()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in self.documents])

    def search(self, question, top_k=3):
        if not isinstance(question, str) or not question.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).ravel()
        indices = scores.argsort()[::-1][:top_k]
        return [{**self.documents[i], "score": float(scores[i])} for i in indices]

    def answer(self, question, threshold=0.15):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        results = self.search(question, top_k=1)
        if not results or results[0]["score"] < threshold:
            return {"answer": "I could not find a reliable answer in the provided documents.", "source": None, "score": 0.0}
        best = results[0]
        return {"answer": best["text"], "source": best["source"], "score": best["score"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()
    result = Retriever().answer(args.question, args.threshold)
    print(result["answer"])
    print(f"Source: {result['source'] or 'none'}")
    print(f"Retrieval score: {result['score']:.3f}")


if __name__ == "__main__":
    main()
