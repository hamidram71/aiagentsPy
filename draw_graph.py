#!/usr/bin/env python3
"""Print the compiled graph as Mermaid, and optionally save a PNG.

    python draw_graph.py            # mermaid text, paste into mermaid.live
    python draw_graph.py --png      # outputs/graph.png (needs network access)

Drawing the graph is the fastest way to catch a wiring mistake: a node with
no incoming edge, or a worker that never returns to the supervisor, is
obvious in the picture and invisible in the code.
"""

from __future__ import annotations

import sys

from src.config import OUTPUT_DIR
from src.graph import build_graph


def main() -> int:
    graph = build_graph()
    print(graph.get_graph().draw_mermaid())

    if "--png" in sys.argv:
        path = OUTPUT_DIR / "graph.png"
        try:
            path.write_bytes(graph.get_graph().draw_mermaid_png())
            print(f"\nsaved {path}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(
                f"\ncould not render PNG ({exc}). The Mermaid text above still "
                "works at https://mermaid.live",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
