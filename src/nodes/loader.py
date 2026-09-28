"""First node: read the CSV and describe it. No LLM call here."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.dataset import load_df, profile_df, profile_to_text
from src.errors import safe_node
from src.state import AgentState


@safe_node("loader")
def loader_node(state: AgentState) -> dict:
    """Load the file, profile it, and seed the conversation.

    The seeded HumanMessage is what every later node sees as "the task", so it
    carries both the question and the data profile. Doing this once means no
    other node has to re-describe the dataset.
    """
    csv_path = state["csv_path"]
    df = load_df(csv_path)
    profile = profile_df(df)

    seed = (
        f"Dataset: {csv_path}\n"
        f"Question: {state['question']}\n\n"
        f"Data profile:\n{profile_to_text(profile)}"
    )

    return {
        "profile": profile,
        "messages": [HumanMessage(content=seed)],
    }
