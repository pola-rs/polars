import pytest

import polars as pl
from polars.testing import assert_frame_equal
from sample_plugin import pig_latinnify, pig_latinnify_deprecated


def test_pig_latinnify() -> None:
    """test_pig_latinnify."""
    df = pl.DataFrame(
        {
            "english": ["this", "is", "not", "pig", "latin"],
        }
    )
    result = df.with_columns(pig_latin=pig_latinnify("english"))
    expected = pl.DataFrame(
        {
            "english": ["this", "is", "not", "pig", "latin"],
            "pig_latin": ["histay", "siay", "otnay", "igpay", "atinlay"],
        }
    )
    assert_frame_equal(result, expected)


def test_pig_latinnify_deprecated() -> None:
    """test_pig_latinnify."""
    df = pl.DataFrame(
        {
            "english": ["this", "is", "not", "pig", "latin"],
        }
    )
    with pytest.deprecated_call():
        result = df.with_columns(pig_latin=pig_latinnify_deprecated("english"))
    expected = pl.DataFrame(
        {
            "english": ["this", "is", "not", "pig", "latin"],
            "pig_latin": ["histay", "siay", "otnay", "igpay", "atinlay"],
        }
    )
    assert_frame_equal(result, expected)
