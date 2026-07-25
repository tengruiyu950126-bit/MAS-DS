"""Bounded, deterministic CSV ingestion for the in-memory application."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


class CSVIngestionError(ValueError):
    """Stable user-facing error for rejected in-memory CSV inputs."""


@dataclass(frozen=True)
class CSVLimits:
    max_upload_bytes: int = 10_000_000
    max_rows: int = 100_000
    max_columns: int = 200
    max_header_characters: int = 128
    max_cell_characters: int = 10_000
    preview_rows: int = 50
    max_diff_rows: int = 1_000
    max_audit_rows: int = 100


DEFAULT_CSV_LIMITS = CSVLimits()


def load_csv_bytes(
    content: bytes,
    *,
    limits: CSVLimits = DEFAULT_CSV_LIMITS,
) -> pd.DataFrame:
    """Parse a UTF-8 CSV only after enforcing in-memory safety limits."""
    if not content:
        raise CSVIngestionError("The uploaded CSV is empty.")
    if len(content) > limits.max_upload_bytes:
        raise CSVIngestionError(
            "The uploaded CSV exceeds the in-memory size limit. "
            "Use the chunked CLI for larger files."
        )
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CSVIngestionError(
            "The uploaded CSV must use UTF-8 encoding."
        ) from exc

    try:
        header = next(csv.reader(io.StringIO(text)))
    except (csv.Error, StopIteration) as exc:
        raise CSVIngestionError("The uploaded CSV has an invalid header.") from exc
    if not header or all(not value.strip() for value in header):
        raise CSVIngestionError("The uploaded CSV must contain named columns.")
    normalized_headers = [value.strip() for value in header]
    if any(not value for value in normalized_headers):
        raise CSVIngestionError("CSV column names must not be empty.")
    if any(len(value) > limits.max_header_characters for value in normalized_headers):
        raise CSVIngestionError("A CSV column name exceeds the length limit.")
    if len(set(normalized_headers)) != len(normalized_headers):
        raise CSVIngestionError("Duplicate CSV column names are not supported.")
    if len(normalized_headers) > limits.max_columns:
        raise CSVIngestionError("The uploaded CSV exceeds the column limit.")

    try:
        dataframe = pd.read_csv(
            io.StringIO(text),
            nrows=limits.max_rows + 1,
        )
    except (EmptyDataError, ParserError, UnicodeError, ValueError) as exc:
        raise CSVIngestionError(
            "The uploaded file is not a usable CSV."
        ) from exc
    if len(dataframe) > limits.max_rows:
        raise CSVIngestionError(
            "The uploaded CSV exceeds the row limit. "
            "Use the chunked CLI for larger files."
        )
    if len(dataframe.columns) > limits.max_columns:
        raise CSVIngestionError("The uploaded CSV exceeds the column limit.")

    for column in dataframe.columns:
        values = dataframe[column].dropna()
        if values.map(lambda value: len(str(value)) > limits.max_cell_characters).any():
            raise CSVIngestionError("A CSV cell exceeds the length limit.")
    return dataframe
