import pandas as pd

from tools.profiler import profile_dataframe


def test_profile_dataframe_counts_missing_and_duplicates() -> None:
    dataframe = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Alice"],
            "age": [25.0, None, 25.0],
        }
    )

    profile = profile_dataframe(dataframe)

    assert profile.rows == 3
    assert profile.columns == 2
    assert profile.duplicate_rows == 1
    assert profile.column_profiles["age"].missing_count == 1
    assert profile.column_profiles["age"].missing_ratio == 1 / 3
