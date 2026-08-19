"""LangGraph orchestration: sequential Data Retriever -> Report Generator workflow."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents import build_data_retriever_agent, run_data_retriever, run_report_generator


class AgentState(TypedDict):
    query: str
    retrieved_snippets: str
    final_answer: str


def build_graph(llm):
    retriever_agent = build_data_retriever_agent(llm)

    def data_retriever_node(state: AgentState) -> dict:
        snippets = run_data_retriever(retriever_agent, state["query"])
        return {"retrieved_snippets": snippets}

    def report_generator_node(state: AgentState) -> dict:
        answer = run_report_generator(llm, state["query"], state["retrieved_snippets"])
        return {"final_answer": answer}

    graph = StateGraph(AgentState)
    graph.add_node("data_retriever", data_retriever_node)
    graph.add_node("report_generator", report_generator_node)
    graph.add_edge(START, "data_retriever")
    graph.add_edge("data_retriever", "report_generator")
    graph.add_edge("report_generator", END)

    return graph.compile()
