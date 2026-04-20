from langchain_core.messages import HumanMessage

from src.graph import build_graph


def run(user_input: str) -> str:
    graph = build_graph()
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello, what can you do?"
    print(run(prompt))
