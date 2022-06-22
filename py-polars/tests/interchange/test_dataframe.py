from polars import DataFrame
from polars.interchange import from_dataframe

# TODO: test data
DF = DataFrame()


def test_dataframe() -> None:
    assert hasattr(DF, "__dataframe__")

    ix_df = DF.__dataframe__()

    assert not ix_df.metadata()
    # assert ix_df.num_columns == 0
    # assert ix_df.num_rows == 0
    # assert ix_df.num_chunks == 0
    # assert not ix_df.column_names
    # assert ix_df.get_column()
    # assert ix_df.get_column_by_name()
    # assert ix_df.get_columns()
    # assert ix_df.select_columns
    # assert ix_df.select_columns_by_name
    # assert ix_df.get_chunks()


def test_from_dataframe() -> None:
    # assert from_dataframe(DataFrame())
    from_dataframe
