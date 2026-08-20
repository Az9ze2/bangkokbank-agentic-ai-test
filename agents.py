"""Defines the Data Retriever agent and the Report Generator agent."""

from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

from retrieval_tool import search_knowledge_base

RETRIEVER_SYSTEM_PROMPT = (
    "You are the Data Retriever agent. You are an expert in information "
    "retrieval, not in answering questions. Call the search_knowledge_base "
    "tool exactly once, passing the user's message to it verbatim and "
    "unmodified as the query argument (do not rephrase, shorten, or split it). "
    "Once the tool returns results, respond with ONLY that raw tool output, "
    "verbatim, and nothing else. Do not summarize, explain, or answer the "
    "question yourself."
)

GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Report Generator agent, an expert writer and "
            "synthesizer. Using ONLY the provided information snippets, write "
            "a comprehensive, well-formatted, non-redundant answer to the "
            "user's query. Never repeat the same point or bullet more than "
            "once. If the snippets do not contain enough information to "
            "answer, say so in 1-2 concise sentences — do not produce an "
            "open-ended list of clarifying questions or requested details.",
        ),
        (
            "human",
            "User query: {query}\n\nRetrieved snippets:\n{snippets}",
        ),
    ]
)


def build_data_retriever_agent(llm):
    return create_react_agent(llm, tools=[search_knowledge_base], prompt=RETRIEVER_SYSTEM_PROMPT)


def run_data_retriever(agent, query: str) -> str:
    result = agent.invoke({"messages": [("user", query)]})
    tool_outputs = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    if tool_outputs:
        unique_outputs = list(dict.fromkeys(tool_outputs))
        return "\n\n---\n\n".join(unique_outputs)
    return result["messages"][-1].content


def run_report_generator(llm, query: str, snippets: str) -> str:
    messages = GENERATOR_PROMPT.invoke({"query": query, "snippets": snippets})
    response = llm.invoke(messages)
    return response.content
