import pl from "@polars";

describe("groupby", () => {
  let df: pl.DataFrame;
  beforeEach(() => {
    df = pl.DataFrame({
      "name": ["a", "b", "a", "c", "b"],
      "foo": [1, 3, 3, 5, 7],
      "bar": [2, 4, 4, 6, 8]
    });
  });

  test("aggList", () => {
    const s = pl.Series("a", [1], pl.Int16);
    const actual = df
      .groupBy("name")
      .aggList()
      .sort("name");

    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo": [[1, 3], [3, 7], [5]],
      "bar": [[2, 4], [4, 8], [6]]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("agg:column", () => {
    const actual = df.groupBy("name").agg({
      "foo": "min"
    })
      .sort("name");
    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo": [1, 3, 5]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("agg:columns", () => {
    const actual = df.groupBy("name").agg({
      "foo": "min",
      "bar": "sum"
    })
      .sort("name");
    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo": [1, 3, 5],
      "bar": [6, 12, 6]
    });
    expect(actual).toFrameEqual(expected);
  });

  test("count", () => {
    const actual = df.groupBy("name").count()
      .sort("name");
    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo_count": [2, 2, 1],
      "bar_count": [2, 2, 1]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("first", () => {
    const actual = df.groupBy("name").first()
      .sort("name");
    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo": [1, 3, 5],
      "bar": [2, 4, 6]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("head", () => {
    const actual = df
      .groupBy("name")
      .head(1)
      .sort("name");
    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo": [[1], [3], [5]],
      "bar": [[2], [4], [6]]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("last", () => {
    const actual = df
      .groupBy("name")
      .last()
      .sort("name");

    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo": [3, 7, 5],
      "bar": [4, 8, 6]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("tail", () => {
    const actual = df
      .groupBy("name")
      .tail(1)
      .sort("name");
    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo": [[3], [7], [5]],
      "bar": [[4], [8], [6]]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("max", () => {
    const actual = df
      .groupBy("name")
      .max()
      .sort("name");
    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo": [3, 7, 5],
      "bar": [4, 8, 6]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("mean", () => {
    const actual = df
      .groupBy("name")
      .mean()
      .sort("name");
    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo": [2, 5, 5],
      "bar": [3, 6, 6]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("median", () => {
    const actual = df
      .groupBy("name")
      .median()
      .sort("name");
    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo": [2, 5, 5],
      "bar": [3, 6, 6]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("min", () => {
    const actual = df
      .groupBy("name")
      .min()
      .sort("name");
    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo": [1, 3, 5],
      "bar": [2, 4, 6]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("nUnique", () => {
    const actual = df
      .groupBy("name")
      .nUnique()
      .sort("name");
    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo": [2, 2, 1],
      "bar": [2, 2, 1]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("sum", () => {
    const actual = df
      .groupBy("name")
      .sum()
      .sort("name");
    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo": [4, 10, 5],
      "bar": [6, 12, 6]
    });
    expect(actual).toFrameEqual(expected);
  });
  test.todo("groups");

});
describe("rolling", () => {
  test("rolling", () => {
    let dates = [
      "2020-01-01 13:45:48",
      "2020-01-01 16:42:13",
      "2020-01-01 16:45:09",
      "2020-01-02 18:12:48",
      "2020-01-03 19:45:32",
      "2020-01-08 23:16:43",
    ];

    const df = pl
      .DataFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]})
      .withColumn(pl.col("dt").str.strptime(pl.Datetime));

    const a = pl.col("a");
    const out = df.groupByRolling({indexColumn:"dt", period:"2d"}).agg(
      a.sum().as("sum_a"),
      a.min().as("min_a"),
      a.max().as("max_a"),
    );
    expect(out["sum_a"].toArray()).toEqual([3, 10, 15, 24, 11, 1]);
    expect(out["max_a"].toArray()).toEqual([3, 7, 7, 9, 9, 1]);
    expect(out["min_a"].toArray()).toEqual([3, 3, 3, 3, 2, 1]);
  });
});
