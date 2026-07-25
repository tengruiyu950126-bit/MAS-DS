import pandas as pd

from tools.export_safety import neutralize_spreadsheet_formulas


def test_audit_export_neutralizes_formula_like_text_without_mutating_input() -> None:
    source = pd.DataFrame(
        {"value": ["=1+1", "+cmd", "-2+3", "@SUM(A1)", "plain", 42]}
    )

    safe = neutralize_spreadsheet_formulas(source)

    assert safe["value"].tolist()[:4] == [
        "'=1+1",
        "'+cmd",
        "'-2+3",
        "'@SUM(A1)",
    ]
    assert safe["value"].tolist()[4:] == ["plain", 42]
    assert source.iloc[0, 0] == "=1+1"
