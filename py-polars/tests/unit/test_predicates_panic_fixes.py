"""
Tests for predicate evaluation edge cases (predicates.rs fixes)

These tests cover scenarios where predicate evaluation could fail due to
missing columns or type mismatches. Previously some of these would cause
crashes - now they should raise proper exceptions.
"""
import pytest
import polars as pl
from polars.exceptions import ColumnNotFoundError, ComputeError


class TestColumnStatsEdgeCases:
    """Verify min/max operations handle edge cases correctly."""

    def test_basic_min_max(self):
        """Standard min/max should work as expected."""
        df = pl.DataFrame({"values": [1, 2, 3, 4, 5]})
        
        assert df["values"].min() == 1
        assert df["values"].max() == 5

    def test_empty_series_returns_null(self):
        """An empty series should return null for min/max."""
        df = pl.DataFrame({"col": pl.Series([], dtype=pl.Int64)})
        
        assert df["col"].min() is None
        assert df["col"].max() is None

    def test_all_null_values(self):
        """Series containing only nulls should return null."""
        df = pl.DataFrame({"col": [None, None, None]})
        
        assert df["col"].min() is None
        assert df["col"].max() is None


class TestPredicateColumnLookup:
    """Check that predicates handle missing columns gracefully."""

    def test_filter_on_existing_column(self):
        """Filtering on a valid column should work normally."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        
        result = df.filter(pl.col("x") > 1)
        assert len(result) == 2

    def test_filter_on_missing_column_raises(self):
        """Referencing a non-existent column should raise ColumnNotFoundError."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        
        with pytest.raises(ColumnNotFoundError):
            df.filter(pl.col("does_not_exist") > 1)

    def test_scan_parquet_with_missing_column(self):
        """Scan predicates on missing columns should raise, not crash."""
        import tempfile
        import os
        
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.parquet")
            df.write_parquet(path)
            
            # Valid column filter works
            result = pl.scan_parquet(path).filter(pl.col("a") > 1).collect()
            assert len(result) == 2
            
            # Invalid column should raise
            with pytest.raises(ColumnNotFoundError):
                pl.scan_parquet(path).filter(pl.col("missing_col") > 1).collect()


class TestNullHandling:
    """Ensure null-heavy data doesn't cause issues."""

    def test_null_count_operations(self):
        """Operations on data with many nulls should complete without error."""
        df = pl.DataFrame({
            "sparse": [None, 1, None, 2, None],
            "empty": [None, None, None, None, None]
        })
        
        counts = df.null_count()
        assert counts["sparse"][0] == 3
        assert counts["empty"][0] == 5

    def test_mixed_types_min_max(self):
        """Type-specific min/max should work for different types."""
        int_df = pl.DataFrame({"nums": [1, 2, 3]})
        str_df = pl.DataFrame({"text": ["apple", "banana", "cherry"]})
        
        assert int_df["nums"].min() == 1
        assert str_df["text"].min() == "apple"  # lexicographic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
