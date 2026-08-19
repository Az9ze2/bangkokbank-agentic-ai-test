"""Custom RAG tool: searches knowledge_base.txt for paragraphs relevant to a query."""

import re
import warnings
from pathlib import Path

from langchain_core.tools import tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = Path(__file__).parent / "knowledge_base.txt"
TOP_K = 3
MIN_SIMILARITY = 0.05

_TOKEN_RE = re.compile(r"[a-zA-Z]+")

# Our custom tokenizer stems before sklearn's stop-word filter sees the tokens,
# which triggers a harmless "may be inconsistent" warning; safe to silence.
warnings.filterwarnings("ignore", message="Your stop_words may be inconsistent.*")


def _stem(word: str) -> str:
    """Naive suffix-stripping so simple plurals/verb forms overlap (cats~cat, lives~live)."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokenize(text: str) -> list[str]:
    return [_stem(w.lower()) for w in _TOKEN_RE.findall(text)]


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

    vectorizer = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None, stop_words="english")
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
