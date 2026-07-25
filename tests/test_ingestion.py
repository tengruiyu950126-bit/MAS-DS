import pandas as pd
import pytest

from tools.ingestion import CSVIngestionError, CSVLimits, load_csv_bytes


SMALL_LIMITS = CSVLimits(
    max_upload_bytes=100,
    max_rows=2,
    max_columns=2,
    max_header_characters=8,
    max_cell_characters=5,
)


def test_load_csv_bytes_accepts_bounded_utf8_csv() -> None:
    result = load_csv_bytes(b"name,age\nAnn,20\nBob,30\n", limits=SMALL_LIMITS)

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({"name": ["Ann", "Bob"], "age": [20, 30]}),
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "empty"),
        (b"name,name\nA,B\n", "Duplicate"),
        (b",age\nA,1\n", "must not be empty"),
        (b"longheader,age\nA,1\n", "column name"),
        (b"a,b,c\n1,2,3\n", "column limit"),
        (b"a\n1\n2\n3\n", "row limit"),
        (b"a\n123456\n", "cell exceeds"),
        (b"\xff\xfe\x00", "UTF-8"),
    ],
)
def test_load_csv_bytes_rejects_unsafe_inputs(
    content: bytes,
    message: str,
) -> None:
    with pytest.raises(CSVIngestionError, match=message):
        load_csv_bytes(content, limits=SMALL_LIMITS)


def test_load_csv_bytes_rejects_oversized_payload_before_parsing() -> None:
    with pytest.raises(CSVIngestionError, match="size limit"):
        load_csv_bytes(b"a" * 101, limits=SMALL_LIMITS)
