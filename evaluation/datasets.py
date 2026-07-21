"""Built-in realistic dataset generators for local experiments and demos."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import random

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """A named dataframe plus a short user-facing description."""

    name: str
    description: str
    dataframe: pd.DataFrame


def _repeat(values: list[str], rows: int) -> list[str]:
    return [values[index % len(values)] for index in range(rows)]


def make_demo_dataset(rows: int = 40) -> pd.DataFrame:
    """Small generic table used by the original vertical slice."""
    return pd.DataFrame(
        {
            "age": [20 + index % 35 for index in range(rows)],
            "score": [round(50 + index * 0.75, 2) for index in range(rows)],
            "city": _repeat(
                ["Singapore", "Kuala Lumpur", "Jakarta", "Bangkok"],
                rows,
            ),
            "signup_date": pd.date_range("2025-01-01", periods=rows, freq="D"),
        }
    )


def make_sales_dataset(rows: int = 60) -> pd.DataFrame:
    """Retail-style order table with numeric, categorical, and date columns."""
    regions = ["APAC", "EMEA", "North America", "LATAM"]
    channels = ["online", "retail", "partner"]
    categories = ["hardware", "software", "service", "subscription"]
    return pd.DataFrame(
        {
            "order_id": [f"ORD-{index + 1:05d}" for index in range(rows)],
            "quantity": [1 + index % 7 for index in range(rows)],
            "unit_price": [
                round(19.99 + (index % 13) * 4.75, 2) for index in range(rows)
            ],
            "discount_rate": [round((index % 5) * 0.03, 2) for index in range(rows)],
            "region": _repeat(regions, rows),
            "channel": _repeat(channels, rows),
            "category": _repeat(categories, rows),
            "order_date": pd.date_range("2025-03-01", periods=rows, freq="D"),
        }
    )


def make_customers_dataset(rows: int = 60) -> pd.DataFrame:
    """Customer profile table with demographic and engagement fields."""
    countries = ["Singapore", "Malaysia", "Indonesia", "Thailand", "Vietnam"]
    segments = ["starter", "growth", "enterprise"]
    acquisition = ["organic", "ads", "referral", "event"]
    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{index + 1:05d}" for index in range(rows)],
            "age": [18 + index % 47 for index in range(rows)],
            "annual_income": [
                28000 + (index % 31) * 1750 for index in range(rows)
            ],
            "loyalty_score": [
                round(35 + (index * 1.7) % 60, 2) for index in range(rows)
            ],
            "country": _repeat(countries, rows),
            "segment": _repeat(segments, rows),
            "acquisition_channel": _repeat(acquisition, rows),
            "signup_date": pd.date_range("2024-06-01", periods=rows, freq="3D"),
        }
    )


def make_students_dataset(rows: int = 60) -> pd.DataFrame:
    """Student records table for academic-data preprocessing demos."""
    majors = ["Computer Science", "Data Science", "Business", "Psychology"]
    campuses = ["main", "city", "online"]
    cohorts = ["2023A", "2023B", "2024A", "2024B"]
    return pd.DataFrame(
        {
            "student_id": [f"STU-{index + 1:05d}" for index in range(rows)],
            "attendance_rate": [
                round(0.55 + ((index * 7) % 45) / 100, 2)
                for index in range(rows)
            ],
            "exam_score": [
                round(48 + (index * 2.3) % 50, 2) for index in range(rows)
            ],
            "credits_completed": [12 + (index % 10) * 3 for index in range(rows)],
            "major": _repeat(majors, rows),
            "campus": _repeat(campuses, rows),
            "cohort": _repeat(cohorts, rows),
            "enrollment_date": pd.date_range(
                "2023-08-15",
                periods=rows,
                freq="7D",
            ),
        }
    )


DATASET_BUILDERS: dict[str, Callable[[int], pd.DataFrame]] = {
    "demo": make_demo_dataset,
    "sales": make_sales_dataset,
    "customers": make_customers_dataset,
    "students": make_students_dataset,
}

DATASET_DESCRIPTIONS: dict[str, str] = {
    "demo": "Small generic profile table.",
    "sales": "Retail order table.",
    "customers": "Customer profile and engagement table.",
    "students": "Academic student-record table.",
}


def available_dataset_names() -> list[str]:
    return list(DATASET_BUILDERS)


def load_dataset(name: str, rows: int = 60) -> DatasetSpec:
    try:
        builder = DATASET_BUILDERS[name]
    except KeyError as exc:
        valid = ", ".join(available_dataset_names())
        raise ValueError(f"Unknown dataset {name!r}. Valid datasets: {valid}.") from exc
    return DatasetSpec(
        name=name,
        description=DATASET_DESCRIPTIONS[name],
        dataframe=builder(rows),
    )


def load_datasets(names: Iterable[str], rows: int = 60) -> list[DatasetSpec]:
    return [load_dataset(name, rows=rows) for name in names]


def make_dirty_copy(
    dataframe: pd.DataFrame,
    *,
    missing_fraction: float = 0.04,
    duplicate_fraction: float = 0.04,
    seed: int = 0,
) -> pd.DataFrame:
    """Create a deterministic dirty UI/demo copy without internal row IDs."""
    if dataframe.empty:
        return dataframe.copy(deep=True)

    rng = random.Random(seed)
    dirty = dataframe.copy(deep=True).reset_index(drop=True)
    editable_columns = [
        column
        for column in dirty.columns
        if "id" not in str(column).lower() and dirty[column].notna().any()
    ]
    if editable_columns:
        candidates = [
            (row_index, column)
            for column in editable_columns
            for row_index in dirty.index
            if pd.notna(dirty.at[row_index, column])
        ]
        missing_count = max(1, round(len(candidates) * missing_fraction))
        for row_index, column in rng.sample(
            candidates,
            min(missing_count, len(candidates)),
        ):
            dirty.at[row_index, column] = None

    duplicate_count = max(1, round(len(dirty) * duplicate_fraction))
    duplicate_indices = rng.sample(
        list(dirty.index),
        min(duplicate_count, len(dirty)),
    )
    duplicates = dirty.loc[duplicate_indices].copy(deep=True)
    return pd.concat([dirty, duplicates], ignore_index=True)
