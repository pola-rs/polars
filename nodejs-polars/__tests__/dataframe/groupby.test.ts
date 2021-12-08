import pl from "@polars";

describe("groupby", () => {
  let df;
  beforeEach(() => {
    df = pl.DataFrame([
      ["a", 1, 2],
      ["b", 3, 4],
      ["a", 3, 4],
      ["c", 5, 6],
      ["b", 7, 8],
    ], {columns: ["name", "foo", "bar"]});
  });

  test("aggList", () => {
    const s = pl.Series("a", [1], pl.Int16);
    const actual = df
      .groupBy("name")
      .aggList()
      .sort("name");

    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo_agg_list": [[1, 3], [3, 7], [5]],
      "bar_agg_list": [[2, 4], [4, 8], [6]]
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
      "foo_min": [1, 3, 5]
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
      "foo_min": [1, 3, 5],
      "bar_sum": [6, 12, 6]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("agg:columns:multi_agg", () => {
    const actual = df.groupBy("name").agg({
      "foo": ["min", "first", "last"],
      "bar": "sum"
    })
      .sort("name");
    const expected = pl.DataFrame({
      "name": ["a", "b", "c"],
      "foo_min": [1, 3, 5],
      "foo_first": [1, 3, 5],
      "foo_last": [3, 7, 5],
      "bar_sum": [6, 12, 6]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("count", () => {

  });
});