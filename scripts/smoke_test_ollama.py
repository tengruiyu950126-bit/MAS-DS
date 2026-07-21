"""Run one real local-model planning request without modifying any files."""

from __future__ import annotations

import os

import pandas as pd

from agents.local_llm_cleaning_agent import LocalLLMCleaningAgent
from providers.ollama import OllamaClient


def main() -> None:
    dataframe = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 3],
            "age": [21.0, None, 35.0, 35.0],
            "city": ["Singapore", "Singapore", None, None],
        }
    )
    model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    agent = LocalLLMCleaningAgent(
        OllamaClient(model=model, timeout_seconds=180)
    )
    plan = agent.propose(dataframe)
    print(plan.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
