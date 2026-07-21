"""Profiling expert backed by deterministic dataframe inspection."""

from __future__ import annotations

import pandas as pd

from models.profile import DataProfile
from tools.profiler import profile_dataframe


class ProfilingAgent:
    """Produces a reproducible data profile for downstream experts."""

    name = "profiling_expert"

    def run(self, dataframe: pd.DataFrame) -> DataProfile:
        return profile_dataframe(dataframe)
