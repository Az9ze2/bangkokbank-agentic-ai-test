"""Runs the two-agent RAG system against a few sample queries."""

from dotenv import load_dotenv

from config import get_llm
from graph import build_graph

SAMPLE_QUERIES = [
    "How long do indoor cats typically live and how can I help mine live longer?",
    "Why do cats purr, and what does a slow blink mean?",
    "Which cat breed is hairless and needs regular bathing?",
    "What is the policy on international travel?",  # deliberately not in the KB
]


def main():
    load_dotenv()
    llm = get_llm()
    app = build_graph(llm)

    for query in SAMPLE_QUERIES:
        result = app.invoke({"query": query, "retrieved_snippets": "", "final_answer": ""})
        print("=" * 80)
        print(f"QUERY: {query}")
        print("-" * 80)
        print("RETRIEVED SNIPPETS:")
        print(result["retrieved_snippets"])
        print("-" * 80)
        print("FINAL ANSWER:")
        print(result["final_answer"])
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()
