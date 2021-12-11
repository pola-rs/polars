import pl, {col, cols, lit} from "@polars/index";


describe("lazy functions", () => {
  test("col:string", () => {
    const expected = pl.Series("foo", [1, 2, 3]);
    const other = pl.Series("other", [1, 2, 3]);
    const df = pl.DataFrame([
      expected,
      other
    ]);
    const actual = df.select(col("foo"));
    expect(actual).toFrameEqual(expected.toFrame());
  });
  test("col:string[]", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [4, 5, 6],
      "other": ["a", "b", "c"]
    }).select(col(["foo", "bar"]));
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [4, 5, 6],
    });
    expect(actual).toFrameEqual(expected);
  });

  test("col:series", () => {
    const columnsToSelect = pl.Series(["foo", "bar"]);
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [4, 5, 6],
      "other": ["a", "b", "c"]
    }).select(col(columnsToSelect));
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [4, 5, 6],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("cols", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [4, 5, 6],
      "other": ["a", "b", "c"]
    }).select(cols("foo", "bar"));
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [4, 5, 6],
    });
    expect(actual).toFrameEqual(expected);
  });
  describe("lit", () => {
    test("string", () => {
      const actual = pl.DataFrame({
        "foo": [1, 2, 3],
        "bar": [4, 5, 6],
      }).select(col("foo"), lit("a").as("lit_a"));
      const expected = pl.DataFrame({
        "foo": [1, 2, 3],
        "lit_a": ["a", "a", "a"],
      });
      expect(actual).toFrameEqual(expected);
    });
    test("number", () => {
      const actual = pl.DataFrame({
        "foo": [1, 2, 3],
        "bar": [4, 5, 6],
      }).select(col("foo"), lit(-99).as("number"));
      const expected = pl.DataFrame({
        "foo": [1, 2, 3],
        "number": [-99, -99, -99],
      });
      expect(actual).toFrameEqual(expected);
    });
    test("bigint", () => {
      const actual = pl.DataFrame({
        "foo": [1, 2, 3],
        "bar": [4, 5, 6],
      }).select(col("foo"), lit(999283899189222n).as("bigint"));
      const expected = pl.DataFrame({
        "foo": [1, 2, 3],
        "bigint": [999283899189222n, 999283899189222n, 999283899189222n],
      });
      expect(actual).toFrameEqual(expected);
    });
    test("series", () => {
      const actual = pl.DataFrame({
        "foo": [1, 2, 3],
        "bar": [4, 5, 6],
      }).select(col("foo"), lit(pl.Series(["one", "two", "three"])).as("series:string"));
      const expected = pl.DataFrame({
        "foo": [1, 2, 3],
        "series:string": ["one", "two", "three"],
      });
      expect(actual).toFrameEqual(expected);
    });
  });
  test("arange", () => {
    const df = pl.DataFrame({
      "foo": [1, 1, 1],
    });
    const expected = pl.DataFrame({"foo":  [1, 1]});
    const actual = df.filter(pl.col("foo").gtEq(pl.arange(0, 3)));
    expect(actual).toFrameEqual(expected);
  });
  test.only("argSortBy", () => {
    const df = pl.DataFrame(
      {
        "bools": [false, true, false],
        "bools_nulls": [null, true, false],
        "int": [1, 2, 3],
        "int_nulls": [1, null, 3],
        "bigint": [1n, 2n, 3n],
        "bigint_nulls": [1n, null, 3n],
        "floats": [1.0, 2.0, 3.0],
        "floats_nulls": [1.0, null, 3.0],
        "strings": ["foo", "bar", "ham"],
        "strings_nulls": ["foo", null, "ham"],
        "date": [new Date(), new Date(), new Date()],
        "datetime": [13241324, 12341256, 12341234],
      }
    );

    const actual = df.select(pl.argSortBy(["int_nulls", "floats"], [false, true])).getColumn("int_nulls");
    console.log(actual);
  });

});