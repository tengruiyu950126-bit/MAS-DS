"""Value-free result model for local Streamlit runtime validation."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class StreamlitE2EResult(BaseModel):
    schema_version: str = "1.0"
    tested_at_utc: datetime
    python_version: str
    streamlit_version: str
    server_startup_status: str = "not_run"
    health_check_status: str = "not_run"
    automated_ui_scenarios_attempted: list[str] = Field(default_factory=list)
    passed_scenarios: list[str] = Field(default_factory=list)
    failed_scenarios: list[str] = Field(default_factory=list)
    skipped_scenarios: dict[str, str] = Field(default_factory=dict)
    rollback_verification_result: str
    download_generation_verification: dict[str, bool] = Field(default_factory=dict)
    process_cleanup_result: str = "not_run"
    warnings: list[str] = Field(default_factory=list)
