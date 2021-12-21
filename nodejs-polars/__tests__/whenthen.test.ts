import pl, {col, lit, when} from "@polars";

describe("when", () => {
  test("when(a).then(b).otherwise(c)", () => {
    const expr = when(col("foo").gt(2))
      .then(lit(1))
      .otherwise(lit(-1))
      .as("when");

    const actual = pl.DataFrame({
      "foo": [1, 3, 4],
      "bar": [3, 4, 0]
    }).withColumn(expr);

    const expected = pl.DataFrame({
      "foo": [1, 3, 4],
      "bar": [3, 4, 0],
      "when": [-1, 1, 1]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("when(a).then(b).when(c).then(d).when(e).then(f).otherwise(g)", () => {
    const expr = when(col("foo").eq(1))
      .then(lit("foo=1"))
      .when(col("foo").eq(3))
      .then(lit("foo=3"))
      .when(col("bar").eq(0)
        .and(col("foo").eq(4)))
      .then(lit("bar=0, foo=4"))
      .otherwise(lit("unknown"))
      .as("when");

    const actual = pl.DataFrame({
      "foo": [1, 3, 4, 5],
      "bar": [3, 4, 0, 1]
    }).withColumn(expr);

    const expected = pl.DataFrame({
      "foo": [1, 3, 4, 5],
      "bar": [3, 4, 0, 1],
      "when": ["foo=1", "foo=3", "bar=0, foo=4", "unknown"]
    });
    expect(actual).toFrameEqual(expected);
  });
});
