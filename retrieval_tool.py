"""Custom RAG tool: searches knowledge_base.txt for paragraphs relevant to a query."""

from pathlib import Path

from langchain_core.tools import tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = Path(__file__).parent / "knowledge_base.txt"
TOP_K = 2
MIN_SIMILARITY = 0.05


def _load_paragraphs() -> list[str]:
    text = KB_PATH.read_text(encoding="utf-8")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


@tool
def search_knowledge_base(query: str) -> str:
    """Search knowledge_base.txt and return the text paragraphs most relevant to the query.

    Use this tool to find raw information snippets. Do not answer the user's
    question yourself; only return what this tool gives back.
    """
    paragraphs = _load_paragraphs()
    if not paragraphs:
        return "The knowledge base is empty."

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(paragraphs + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()

    ranked = sorted(
        ((score, para) for score, para in zip(scores, paragraphs)),
        key=lambda item: item[0],
        reverse=True,
    )
    relevant = [para for score, para in ranked[:TOP_K] if score >= MIN_SIMILARITY]

    if not relevant:
        return "No relevant information found in the knowledge base."

    return "\n\n---\n\n".join(relevant)
