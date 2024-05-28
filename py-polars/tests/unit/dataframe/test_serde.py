import pytest

import polars as pl


def test_df_serialize() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).sort("a")
    result = df.serialize()
    expected = '{"columns":[{"name":"a","datatype":"Int64","bit_settings":"SORTED_ASC","values":[1,2,3]},{"name":"b","datatype":"Int64","bit_settings":"","values":[4,5,6]}]}'
    assert result == expected


def test_df_write_json_deprecated() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.deprecated_call():
        result = df.write_json()
    assert result == df.serialize()


def test_df_write_json_pretty_deprecated() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.deprecated_call():
        result = df.write_json(pretty=True)
    assert isinstance(result, str)
