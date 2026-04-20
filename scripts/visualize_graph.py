"""
Render the full agent graph (including subgraph internals) and save as PNG.

Usage:
    python scripts/visualize_graph.py               # saves graph.png
    python scripts/visualize_graph.py --out my.png  # custom filename

How it works:
    LangGraph's get_graph(xray=True) expands subgraphs so you see every node
    across all three agents. draw_mermaid_png() calls the Mermaid.ink API to
    render the diagram — no local graphviz needed.
"""
import argparse
from pathlib import Path

from src.graph import build_graph


def main(out: str) -> None:
    graph = build_graph()

    # xray=True expands subgraph nodes so you see the full topology
    png_bytes = graph.get_graph(xray=True).draw_mermaid_png()

    Path(out).write_bytes(png_bytes)
    print(f"Graph saved to {out}")

    # Also print the Mermaid source (paste into mermaid.live to edit)
    mermaid_src = graph.get_graph(xray=True).draw_mermaid()
    print("\n--- Mermaid source ---")
    print(mermaid_src)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="graph.png")
    args = parser.parse_args()
    main(args.out)
