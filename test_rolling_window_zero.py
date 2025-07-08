#!/usr/bin/env python3
"""
Test case for rolling window functions with window_size=0.
This test ensures that all rolling window functions properly validate that window_size > 0.
"""

import pytest
import polars as pl
from polars.exceptions import InvalidOperationError


def test_rolling_window_size_zero_validation():
    """Test that rolling functions raise an error when window_size=0."""
    s = pl.Series([1, 2, 3, 4, 5])
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    
    # Test Series rolling functions
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        s.rolling_sum(window_size=0)
    
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        s.rolling_mean(window_size=0)
    
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        s.rolling_std(window_size=0)
    
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        s.rolling_min(window_size=0)
    
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        s.rolling_max(window_size=0)
    
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        s.rolling_var(window_size=0)
    
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        s.rolling_median(window_size=0)
    
    # Test expression rolling functions
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        df.select(pl.col("a").rolling_sum(window_size=0)).collect()
    
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        df.select(pl.col("a").rolling_mean(window_size=0)).collect()
    
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        df.select(pl.col("a").rolling_std(window_size=0)).collect()
    
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        df.select(pl.col("a").rolling_min(window_size=0)).collect()
    
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        df.select(pl.col("a").rolling_max(window_size=0)).collect()
    
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        df.select(pl.col("a").rolling_var(window_size=0)).collect()
    
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        df.select(pl.col("a").rolling_median(window_size=0)).collect()
    
    # Test lazy frame rolling functions
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        df.lazy().select(pl.col("a").rolling_sum(window_size=0)).collect()
    
    with pytest.raises(InvalidOperationError, match="`window_size` must be strictly positive, got: 0"):
        df.lazy().select(pl.col("a").rolling_mean(window_size=0)).collect()


def test_rolling_window_size_positive_works():
    """Test that rolling functions work correctly with positive window_size."""
    s = pl.Series([1, 2, 3, 4, 5])
    
    # Test that valid window sizes work
    result = s.rolling_sum(window_size=2)
    expected = pl.Series([None, 3, 5, 7, 9])
    assert result.to_list() == expected.to_list()
    
    result = s.rolling_mean(window_size=2)
    expected = pl.Series([None, 1.5, 2.5, 3.5, 4.5])
    assert result.to_list() == expected.to_list()


if __name__ == "__main__":
    test_rolling_window_size_zero_validation()
    test_rolling_window_size_positive_works()
    print("All tests passed!")
