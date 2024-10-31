import polars as pl


def test_literal_broadcast_array() -> None:
    df = pl.DataFrame({"A": [[0.1, 0.2], [0.3, 0.4]]}).cast(pl.Array(float, 2))

    lit = pl.lit([3, 5], pl.Array(float, 2))
    assert df.select(
        mul=pl.all() * lit,
        div=pl.all() / lit,
        add=pl.all() + lit,
        sub=pl.all() - lit,
        div_=lit / pl.all(),
        add_=lit + pl.all(),
        sub_=lit - pl.all(),
        mul_=lit * pl.all(),
    ).to_dict(as_series=False) == {
        "mul": [[0.30000000000000004, 1.0], [0.8999999999999999, 2.0]],
        "div": [[0.03333333333333333, 0.04], [0.09999999999999999, 0.08]],
        "add": [[3.1, 5.2], [3.3, 5.4]],
        "sub": [[-2.9, -4.8], [-2.7, -4.6]],
        "div_": [[30.0, 25.0], [10.0, 12.5]],
        "add_": [[3.1, 5.2], [3.3, 5.4]],
        "sub_": [[2.9, 4.8], [2.7, 4.6]],
        "mul_": [[0.30000000000000004, 1.0], [0.8999999999999999, 2.0]],
    }
