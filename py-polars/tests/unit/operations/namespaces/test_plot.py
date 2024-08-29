import polars as pl


def test_dataframe_plot() -> None:
    # dry-run, check nothing errors
    df = pl.DataFrame(
        {
            "length": [1, 4, 6],
            "width": [4, 5, 6],
            "species": ["setosa", "setosa", "versicolor"],
        }
    )
    df.plot.line(x="length", y="width", color="species").to_json()
    df.plot.point(x="length", y="width", size="species").to_json()
    df.plot.scatter(x="length", y="width", size="species").to_json()
    df.plot.bar(x="length", y="width", color="species").to_json()
    df.plot.area(x="length", y="width", color="species").to_json()


def test_series_plot() -> None:
    # dry-run, check nothing errors
    s = pl.Series("a", [1, 4, 4, 4, 7, 2, 5, 3, 6])
    s.plot.kde().to_json()
    s.plot.hist().to_json()
    s.plot.line().to_json()
    s.plot.point().to_json()


def test_empty_dataframe() -> None:
    pl.DataFrame({"a": [], "b": []}).plot.point(x="a", y="b")


def test_nameless_series() -> None:
    pl.Series([1, 2, 3]).plot.kde().to_json()
