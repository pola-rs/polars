import polars as pl


def test_show_graph_ir() -> None:
    # only test raw output, otherwise we need graphviz and matplotlib
    ldf = pl.LazyFrame(
        {
            "a": ["a", "b", "a", "b", "b", "c"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )
    query = ldf.group_by("a", maintain_order=True).agg(pl.all().sum()).sort("a")
    out = query.show_graph(raw_output=True, plan_stage="ir")
    assert isinstance(out, str)


def test_show_graph_phys() -> None:
    # only test raw output, otherwise we need graphviz and matplotlib
    ldf = pl.LazyFrame(
        {
            "a": ["a", "b", "a", "b", "b", "c"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )
    query = ldf.group_by("a", maintain_order=True).agg(pl.all().sum()).sort("a")
    out = query.show_graph(raw_output=True, plan_stage="phys", engine="streaming")
    assert isinstance(out, str)
