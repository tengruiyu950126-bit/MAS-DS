"""Independent, deterministic ground truth for orchestration evaluation.

Artifacts produced from these scenarios contain metadata only; dataframe values
never leave this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from models.data_contract import ColumnContract, DataContract
from models.policy import PreprocessingPolicy


EXPERTS = (
    "DuplicateExpert", "MissingValueExpert", "NumericExpert",
    "CategoricalExpert", "TextExpert", "DatetimeExpert",
)


@dataclass(frozen=True)
class RoutingScenario:
    scenario_id: str
    family: str
    seed: int
    dataframe: pd.DataFrame
    problems: tuple[str, ...]
    affected_columns: tuple[str, ...]
    expected_routes: frozenset[tuple[str, str]]
    expected_operations: frozenset[tuple[str, str]]
    protected_columns: tuple[str, ...] = ()
    expected_exclusions: tuple[str, ...] = ()
    expected_safe_behavior: str = "propose bounded deterministic repairs"
    multiple_experts: bool = False
    conflicting_signals: bool = False
    clean: bool = False
    contract: DataContract | None = None

    def policy(self) -> PreprocessingPolicy:
        return PreprocessingPolicy(protected_columns=list(self.protected_columns))

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": "1.0", "scenario_id": self.scenario_id,
            "family": self.family, "seed": self.seed,
            "rows": len(self.dataframe), "columns": len(self.dataframe.columns),
            "problem_types": "|".join(self.problems),
            "affected_columns": "|".join(self.affected_columns),
            "expected_experts": "|".join(sorted({e for _, e in self.expected_routes})),
            "protected_columns": "|".join(self.protected_columns),
            "expected_exclusions": "|".join(self.expected_exclusions),
            "expected_safe_behavior": self.expected_safe_behavior,
            "multiple_experts": self.multiple_experts,
            "conflicting_signals": self.conflicting_signals,
            "clean": self.clean,
            "contract_active": self.contract is not None,
        }


def _base(seed: int, rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "row_id": [f"R{i:04d}" for i in range(rows)],
        "amount": rng.normal(50, 4, rows).round(2),
        "category": np.resize(np.array(["alpha", "beta", "gamma"], dtype=object), rows),
        "notes": np.resize(np.array(["short note", "reviewed note", "plain note", "stable note"], dtype=object), rows),
        "event_date": pd.date_range("2025-01-01", periods=rows),
    })


def _scenario(family: str, seed: int) -> RoutingScenario:
    df = _base(seed)
    rid = lambda column, expert: (column, expert)
    kwargs: dict[str, object] = {}
    routes: set[tuple[str, str]] = set()
    ops: set[tuple[str, str]] = set()
    problems: tuple[str, ...]
    affected: tuple[str, ...]

    if family == "duplicate_rows_only":
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True); problems=("duplicate_rows",); affected=("*",)
        routes.add(rid("*", "DuplicateExpert")); ops.add(("*", "drop_duplicates"))
    elif family == "missing_numeric":
        df.loc[[1, 7], "amount"] = np.nan; problems=("missing_numeric",); affected=("amount",)
        routes.add(rid("amount", "MissingValueExpert")); ops.add(("amount", "fill_median"))
    elif family == "missing_categorical":
        df.loc[[2, 8], "category"] = None; problems=("missing_categorical",); affected=("category",)
        routes.add(rid("category", "MissingValueExpert")); ops.add(("category", "fill_mode"))
    elif family == "numeric_outliers":
        df.loc[3, "amount"] = 5000; problems=("numeric_outlier",); affected=("amount",)
        routes.add(rid("amount", "NumericExpert")); ops.add(("amount", "flag_outliers_iqr"))
    elif family == "numeric_strings":
        df["amount"] = df["amount"].map(lambda x: f"{x:.2f}"); problems=("numeric_text",); affected=("amount",)
        routes.add(rid("amount", "NumericExpert")); ops.add(("amount", "convert_numeric"))
    elif family == "invalid_categorical":
        df.loc[0, "category"] = "Alpha"; problems=("category_case",); affected=("category",)
        routes.add(rid("category", "CategoricalExpert")); ops.add(("category", "normalize_case"))
    elif family == "high_cardinality_categorical":
        df["category"] = [f"label_{i:03d}" for i in range(len(df))]; problems=("high_cardinality",); affected=("category",)
        routes.add(rid("category", "TextExpert"))
        kwargs["expected_safe_behavior"] = "route for bounded text analysis without inventing categories"
    elif family == "text_whitespace":
        df.loc[0, "notes"] = "  padded note  "; problems=("text_whitespace",); affected=("notes",)
        routes.add(rid("notes", "TextExpert")); ops.add(("notes", "strip_whitespace"))
    elif family == "text_length":
        df.loc[0, "notes"] = "x"; problems=("text_length",); affected=("notes",)
        routes.add(rid("notes", "TextExpert")); kwargs["expected_safe_behavior"] = "detect text concern but avoid destructive rewriting"
    elif family == "datetime_parsing":
        df["event_date"] = df["event_date"].dt.strftime("%Y-%m-%d")
        problems=("datetime_text",); affected=("event_date",); routes.add(rid("event_date", "DatetimeExpert")); ops.add(("event_date", "convert_datetime"))
    elif family == "invalid_datetime":
        df["event_date"] = df["event_date"].dt.strftime("%Y-%m-%d")
        df.loc[0, "event_date"] = "not-a-date"; problems=("unsafe_datetime",); affected=("event_date",)
        kwargs["expected_safe_behavior"] = "leave unchanged because parsing is not lossless"; kwargs["conflicting_signals"] = True
    elif family == "mixed_missing_numeric":
        df.loc[[1, 2], "amount"] = np.nan; df.loc[3, "amount"] = 5000; problems=("missing_numeric","numeric_outlier"); affected=("amount",)
        routes |= {rid("amount","MissingValueExpert"), rid("amount","NumericExpert")}; ops |= {("amount","fill_median"),("amount","flag_outliers_iqr")}; kwargs["multiple_experts"] = True
    elif family == "mixed_categorical_text":
        df.loc[0,"category"]="Alpha"; df.loc[0,"notes"]=" padded "; problems=("category_case","text_whitespace"); affected=("category","notes")
        routes |= {rid("category","CategoricalExpert"),rid("notes","TextExpert")}; ops |= {("category","normalize_case"),("notes","strip_whitespace")}; kwargs["multiple_experts"] = True
    elif family == "mixed_numeric_datetime":
        df["amount"] = df["amount"].astype(str); df["event_date"] = df["event_date"].dt.strftime("%Y-%m-%d"); problems=("numeric_text","datetime_text"); affected=("amount","event_date")
        routes |= {rid("amount","NumericExpert"),rid("event_date","DatetimeExpert")}; ops |= {("amount","convert_numeric"),("event_date","convert_datetime")}; kwargs["multiple_experts"] = True
    elif family == "all_six_families":
        df.loc[0,"amount"]=5000; df.loc[1,"amount"]=np.nan; df.loc[0,"category"]="Alpha"; df.loc[0,"notes"]=" padded "; df["event_date"] = df["event_date"].dt.strftime("%Y-%m-%d"); df=pd.concat([df,df.iloc[[2]]],ignore_index=True)
        problems=("duplicates","missing","numeric","categorical","text","datetime"); affected=("*","amount","category","notes","event_date")
        routes |= {rid("*","DuplicateExpert"),rid("amount","MissingValueExpert"),rid("amount","NumericExpert"),rid("category","CategoricalExpert"),rid("notes","TextExpert"),rid("event_date","DatetimeExpert")}
        ops |= {("*","drop_duplicates"),("amount","fill_median"),("amount","flag_outliers_iqr"),("category","normalize_case"),("notes","strip_whitespace"),("event_date","convert_datetime")}; kwargs["multiple_experts"] = True
    elif family == "protected_identifier":
        df.loc[0,"row_id"]=" padded-id "; problems=("protected_identifier",); affected=("row_id",)
        kwargs.update(protected_columns=("row_id",), expected_exclusions=("row_id",), expected_safe_behavior="exclude protected identifier from mutation")
    elif family == "contract_protected":
        df.loc[0,"category"]=" Alpha "; problems=("contract_protected",); affected=("category",)
        kwargs.update(expected_exclusions=("category",), contract=DataContract(name="evaluation contract",protected_columns=["category"],columns={"category":ColumnContract(nullable=False)}), expected_safe_behavior="exclude contract-protected column from mutation")
    elif family == "ambiguous_column":
        df["maybe_value"]=["1", "two"] * 20; problems=("ambiguous_type",); affected=("maybe_value",)
        kwargs.update(conflicting_signals=True, expected_safe_behavior="leave ambiguous mixed-type values unchanged")
    elif family == "clean_control":
        # Protect semantic identifiers; remaining columns are clean controls.
        problems=(); affected=(); kwargs.update(clean=True, protected_columns=("row_id",), expected_safe_behavior="produce no mutating proposals")
    elif family == "unsupported_unsafe":
        # Tuple objects are hashable for profiling but are not a supported
        # semantic cleaning type, so the safe expectation remains no mutation.
        df["payload"]=[("opaque", i) for i in range(len(df))]; problems=("unsupported_object",); affected=("payload",)
        kwargs["expected_safe_behavior"] = "leave unsupported object values unchanged"
    else:
        raise ValueError(f"Unknown scenario family: {family}")
    return RoutingScenario(
        scenario_id=f"{family}__seed_{seed}", family=family, seed=seed,
        dataframe=df, problems=problems, affected_columns=affected,
        expected_routes=frozenset(routes), expected_operations=frozenset(ops), **kwargs,
    )


SCENARIO_FAMILIES = (
    "duplicate_rows_only", "missing_numeric", "missing_categorical", "numeric_outliers",
    "numeric_strings", "invalid_categorical", "high_cardinality_categorical",
    "text_whitespace", "text_length", "datetime_parsing", "invalid_datetime",
    "mixed_missing_numeric", "mixed_categorical_text", "mixed_numeric_datetime",
    "all_six_families", "protected_identifier", "contract_protected", "ambiguous_column",
    "clean_control", "unsupported_unsafe",
)


def generate_scenarios(seeds: Iterable[int], *, quick: bool = False) -> list[RoutingScenario]:
    families = SCENARIO_FAMILIES[:8] if quick else SCENARIO_FAMILIES
    scenarios = [_scenario(family, int(seed)) for seed in seeds for family in families]
    if not quick:
        scenarios.extend(_bundled_public_scenarios(seeds))
    return scenarios


def _bundled_public_scenarios(seeds: Iterable[int]) -> list[RoutingScenario]:
    """Add small, local scikit-learn controls when the optional package exists."""
    try:
        from evaluation.real_dataset_benchmark import load_public_dataset, sklearn_available
        if not sklearn_available():
            return []
    except ImportError:
        return []
    result=[]
    for seed in seeds:
        iris=load_public_dataset("iris",rows=80).dataframe
        numeric=next(c for c in iris.columns if c!="public_row_id" and pd.api.types.is_numeric_dtype(iris[c]))
        iris.loc[int(seed)%len(iris),numeric]=np.nan
        result.append(RoutingScenario(
            scenario_id=f"bundled_iris_missing__seed_{seed}",family="bundled_iris_missing",seed=int(seed),dataframe=iris,
            problems=("missing_numeric",),affected_columns=(numeric,),expected_routes=frozenset({(numeric,"MissingValueExpert")}),
            expected_operations=frozenset({(numeric,"fill_median")}),protected_columns=("public_row_id",),expected_exclusions=("public_row_id",),
            expected_safe_behavior="repair injected missing value while preserving bundled public data",
        ))
        wine=load_public_dataset("wine",rows=80).dataframe
        wine=pd.concat([wine,wine.iloc[[int(seed)%len(wine)]]],ignore_index=True)
        result.append(RoutingScenario(
            scenario_id=f"bundled_wine_duplicate__seed_{seed}",family="bundled_wine_duplicate",seed=int(seed),dataframe=wine,
            problems=("duplicate_rows",),affected_columns=("*",),expected_routes=frozenset({("*","DuplicateExpert")}),
            expected_operations=frozenset({("*","drop_duplicates")}),protected_columns=("public_row_id",),expected_exclusions=("public_row_id",),
            expected_safe_behavior="remove injected duplicate while preserving bundled public data",
        ))
    return result
