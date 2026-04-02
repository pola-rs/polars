import altair as alt

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


def test_dataframe_plot_tooltip() -> None:
    df = pl.DataFrame(
        {
            "length": [1, 4, 6],
            "width": [4, 5, 6],
            "species": ["setosa", "setosa", "versicolor"],
        }
    )
    result = df.plot.line(x="length", y="width", color="species").to_dict()
    assert result["mark"]["tooltip"] is True
    result = df.plot.line(
        x="length", y="width", color="species", tooltip=["length", "width"]
    ).to_dict()
    assert result["encoding"]["tooltip"] == [
        {"field": "length", "type": "quantitative"},
        {"field": "width", "type": "quantitative"},
    ]


def test_series_plot() -> None:
    # dry-run, check nothing errors
    s = pl.Series("a", [1, 4, 4, 4, 7, 2, 5, 3, 6])
    s.plot.kde().to_json()
    s.plot.hist().to_json()
    s.plot.line().to_json()
    s.plot.point().to_json()


def test_series_plot_tooltip() -> None:
    s = pl.Series("a", [1, 4, 4, 4, 7, 2, 5, 3, 6])
    result = s.plot.line().to_dict()
    assert result["mark"]["tooltip"] is True
    result = s.plot.line(tooltip=["a"]).to_dict()
    assert result["encoding"]["tooltip"] == [{"field": "a", "type": "quantitative"}]


def test_empty_dataframe() -> None:
    pl.DataFrame({"a": [], "b": []}).plot.point(x="a", y="b")


def test_nameless_series() -> None:
    pl.Series([1, 2, 3]).plot.kde().to_json()


def test_x_with_axis_18830() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result = df.plot.line(x=alt.X("a", axis=alt.Axis(labelAngle=-90))).to_dict()
    assert result["mark"]["tooltip"] is True


def test_errorbar_19787() -> None:
    df = pl.DataFrame({"A": [0, 1, 2], "B": [10, 11, 12], "C": [1, 2, 3]})
    result = df.plot.errorbar(x="A", y="B", yError="C").to_dict()
    assert "tooltip" not in result["encoding"]
    result = df["A"].plot.errorbar().to_dict()
    assert "tooltip" not in result["encoding"]
