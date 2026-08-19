# Agentic AI Programming Test: Data Retriever + Report Generator

A two-agent RAG system built with **LangGraph**, orchestrated as a sequential
workflow: `Data Retriever -> Report Generator`.

## Architecture

- **`knowledge_base.txt`** — local knowledge base (general cat facts), one
  topic per paragraph.
- **`retrieval_tool.py`** — custom tool `search_knowledge_base` that ranks
  the knowledge base paragraphs against the query using TF-IDF + cosine
  similarity (a lightweight, dependency-free form of semantic search) and
  returns the top matching raw text chunks.
- **`agents.py`**
  - **Data Retriever agent** — a LangGraph ReAct agent (`create_react_agent`)
    bound to `search_knowledge_base`. It is instructed to only call the tool
    and relay its raw output, never to answer the question itself.
  - **Report Generator agent** — a plain LLM call (no tools) that synthesizes
    the retrieved snippets into a single, well-formatted, non-redundant
    answer.
- **`graph.py`** — wires the two agents into a `StateGraph` with the
  sequential edge `START -> data_retriever -> report_generator -> END`,
  passing the retriever's output as the generator's input.
- **`main.py`** — runs the compiled graph against a few sample queries and
  prints the retrieved snippets and final answer for each.

## Setup

1. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in the Azure OpenAI credentials.
   The `gpt-5-mini` subscription key must be requested from
   kanit.mekritthikrai@bangkokbank.com; the endpoint and deployment name are
   already filled in.

## Run

```
python main.py
```

This runs the graph against several sample queries (including one that is
deliberately absent from the knowledge base, to verify the "no relevant
information" path) and prints the retrieved snippets and final answer for
each. Save terminal screenshots into `screenshots/` for submission.
