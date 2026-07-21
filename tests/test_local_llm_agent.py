import json
import urllib.error

import pandas as pd
import pytest

from agents.local_llm_cleaning_agent import (
    LocalLLMCleaningAgent,
    LocalLLMPlanningError,
)
from models.policy import PreprocessingPolicy
from providers.ollama import OllamaClient, OllamaError


class FakeClient:
    model = "fake-local-model"

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_user_prompt = ""
        self.last_schema: dict | None = None

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict | None = None,
    ) -> dict:
        assert "Allowed operations only" in system_prompt
        assert json_schema is not None
        assert "steps" in json_schema["properties"]
        self.last_user_prompt = user_prompt
        self.last_schema = json_schema
        return self.payload


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def valid_payload() -> dict:
    return {
        "steps": [
            {
                "column": "age",
                "operation": "fill_median",
                "reason": "One numeric value is missing.",
                "confidence": 0.9,
            }
        ]
    }


def test_local_llm_agent_returns_validated_plan() -> None:
    client = FakeClient(valid_payload())
    dataframe = pd.DataFrame({"age": [10.0, None, 30.0]})

    plan = LocalLLMCleaningAgent(client).propose(dataframe)

    assert plan.steps[0].operation == "fill_median"
    evidence = client.last_user_prompt.split("DATASET EVIDENCE:\n", 1)[1]
    prompt_payload = json.loads(evidence)
    assert prompt_payload["profile"]["rows"] == 3
    assert len(prompt_payload["sample_rows"]) == 3
    variants = client.last_schema["properties"]["steps"]["items"]["oneOf"]
    column_operation_schema = variants[1]
    assert column_operation_schema["properties"]["column"]["enum"] == ["age"]
    allowed = column_operation_schema["properties"]["operation"]["enum"]
    assert "parse_numeric_text" in allowed
    assert "strip_whitespace" in allowed
    assert "normalize_case" in allowed
    assert "normalize_category_typos" in allowed
    assert "flag_outliers_iqr" in allowed


def test_local_llm_agent_limits_sample_rows() -> None:
    client = FakeClient({"steps": []})
    dataframe = pd.DataFrame({"value": range(100)})

    LocalLLMCleaningAgent(client, sample_rows=4).propose(dataframe)

    evidence = client.last_user_prompt.split("DATASET EVIDENCE:\n", 1)[1]
    assert len(json.loads(evidence)["sample_rows"]) == 4


def test_local_llm_agent_rejects_unknown_column() -> None:
    payload = valid_payload()
    payload["steps"][0]["column"] = "imaginary"

    with pytest.raises(LocalLLMPlanningError, match="unknown column"):
        LocalLLMCleaningAgent(FakeClient(payload)).propose(
            pd.DataFrame({"age": [1, 2]})
        )


def test_local_llm_agent_rejects_duplicate_steps() -> None:
    payload = valid_payload()
    payload["steps"].append(dict(payload["steps"][0]))

    with pytest.raises(LocalLLMPlanningError, match="Duplicate plan step"):
        LocalLLMCleaningAgent(FakeClient(payload)).propose(
            pd.DataFrame({"age": [1, None]})
        )


def test_local_llm_agent_filters_steps_disallowed_by_policy() -> None:
    payload = {
        "steps": [
            {
                "column": "city",
                "operation": "strip_whitespace",
                "reason": "Whitespace exists.",
                "confidence": 0.9,
            }
        ]
    }
    agent = LocalLLMCleaningAgent(
        FakeClient(payload),
        policy=PreprocessingPolicy(protected_columns=["city"]),
    )

    plan = agent.propose(pd.DataFrame({"city": [" Singapore "]}))

    assert plan.steps == []


def test_ollama_client_parses_nested_json(monkeypatch) -> None:
    response = {
        "message": {"content": json.dumps(valid_payload())},
    }

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://localhost:11434/api/chat"
        assert timeout == 5
        request_body = json.loads(request.data.decode("utf-8"))
        assert request_body["think"] is False
        assert request_body["options"]["num_ctx"] == 4096
        assert request_body["format"]["required"] == ["steps"]
        return FakeHTTPResponse(response)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OllamaClient(model="local-test", timeout_seconds=5)

    schema = {
        "type": "object",
        "properties": {"steps": {"type": "array"}},
        "required": ["steps"],
    }
    assert client.generate_json("system", "user", schema) == valid_payload()


def test_ollama_client_reports_connection_failure(monkeypatch) -> None:
    def failing_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)

    with pytest.raises(OllamaError, match="Cannot reach local Ollama"):
        OllamaClient(model="missing-model").generate_json("system", "user")


def test_ollama_client_rejects_invalid_envelope(monkeypatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeHTTPResponse({"unexpected": True}),
    )

    with pytest.raises(OllamaError, match="invalid JSON"):
        OllamaClient(model="local-test").generate_json("system", "user")
