import pl from "@polars";

describe("lazyframe", () => {
  test("columns", () => {
    const df = pl.DataFrame({
      "foo": [1, 2],
      "bar": ["a", "b"]
    }).lazy();
    const actual = df.columns;
    expect(actual).toEqual(["foo", "bar"]);
  });
  test("collectSync", () => {
    const expected = pl.DataFrame({
      "foo": [1, 2],
      "bar": ["a", "b"]
    });
    const actual = expected.lazy().collectSync();
    expect(actual).toFrameEqual(expected);
  });
  test("collect", async () => {
    const expected = pl.DataFrame({
      "foo": [1, 2],
      "bar": ["a", "b"]
    });
    const actual = await expected.lazy().collect();
    expect(actual).toFrameEqual(expected);
  });
  test("describePlan", () => {
    const df = pl.DataFrame({
      "foo": [1, 2],
      "bar": ["a", "b"]
    }).lazy();
    const actual = df.describePlan();
    expect(actual).toEqual(
      `TABLE: ["foo", "bar"]; PROJECT */2 COLUMNS; SELECTION: None\\n
                    PROJECTION: None`);
  });
  test("describeOptimiziedPlan", () => {
    const df = pl.DataFrame({
      "foo": [1, 2],
      "bar": ["a", "b"]
    }).lazy();
    const actual = df.describeOptimizedPlan();
    expect(actual).toEqual(
      `TABLE: ["foo", "bar"]; PROJECT */2 COLUMNS; SELECTION: None\\n
                    PROJECTION: None`);
  });
  test("drop", () =>{
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"],
      "apple": ["a", "b", "c"]
    });
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"],
    });
    const actual = df.lazy()
      .drop("apple")
      .collectSync();
    expect(actual).toFrameEqualIgnoringOrder(expected);
  });
  test("drop:array", () =>{
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"],
      "apple": ["a", "b", "c"]
    });
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
    });
    const actual = df.lazy()
      .drop(["apple", "ham"])
      .collectSync();
    expect(actual).toFrameEqualIgnoringOrder(expected);
  });
  test("drop:rest", () =>{
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"],
      "apple": ["a", "b", "c"]
    });
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
    });
    const actual = df.lazy()
      .drop("apple", "ham")
      .collectSync();
    expect(actual).toFrameEqualIgnoringOrder(expected);
  });
  test("dropDuplicates", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 2, 3],
      "bar": [1, 2, 2, 4],
      "ham": ["a", "d", "d", "c"],
    }).lazy()
      .dropDuplicates()
      .collectSync();
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [1, 2, 4],
      "ham": ["a", "d", "c"],
    });
    expect(actual).toFrameEqualIgnoringOrder(expected);
  });
  test("dropDuplicates:subset", () =>{
    const actual = pl.DataFrame({
      "foo": [1, 2, 2, 2],
      "bar": [1, 2, 2, 2],
      "ham": ["a", "b", "c", "c"],
    }).lazy()
      .dropDuplicates({subset: ["foo", "ham"]})
      .collectSync();
    const expected = pl.DataFrame({
      "foo": [1, 2, 2],
      "bar": [1, 2, 2],
      "ham": ["a", "b", "c"],
    });
    expect(actual).toFrameEqualIgnoringOrder(expected);
  });
  test("dropDuplicates:maintainOrder", () =>{
    const actual = pl.DataFrame({
      "foo": [1, 2, 2, 2],
      "bar": [1, 2, 2, 2],
      "ham": ["a", "b", "c", "d"],
    }).lazy()
      .dropDuplicates({maintainOrder: true, subset: ["foo", "bar"]})
      .collectSync();
    const expected = pl.DataFrame({
      "foo": [1, 2],
      "bar": [1, 2],
      "ham": ["a", "b"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("dropNulls", () => {
    const actual = pl.DataFrame({
      "foo": [1, null, 2, 3],
      "bar": [6.0, .5, 7.0, 8.0],
      "ham": ["a", "d", "b", "c"],
    }).lazy()
      .dropNulls()
      .collectSync();
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"],
    });
    expect(actual).toFrameEqualIgnoringOrder(expected);
  });
  test("dropNulls:array", () => {
    const actual = pl.DataFrame({
      "foo": [1, null, 2, 3],
      "bar": [6.0, .5, null, 8.0],
      "ham": ["a", "d", "b", "c"],
    }).lazy()
      .dropNulls(["foo"])
      .collectSync();
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, null, 8.0],
      "ham": ["a", "b", "c"],
    });
    expect(actual).toFrameEqualIgnoringOrder(expected);
  });
  test("dropNulls:rest", () =>{
    const actual = pl.DataFrame({
      "foo": [1, null, 2, 3],
      "bar": [6.0, .5, null, 8.0],
      "ham": ["a", "d", "b", "c"],
    }).lazy()
      .dropNulls("foo", "bar")
      .collectSync();
    const expected = pl.DataFrame({
      "foo": [1, 3],
      "bar": [6.0, 8.0],
      "ham": ["a", "c"],
    });
    expect(actual).toFrameEqualIgnoringOrder(expected);
  });
  test("explode", () => {
    const actual = pl.DataFrame({
      "letters": ["c", "a"],
      "list_1": [[1, 2], [1, 3]],
    }).lazy()
      .explode("list_1")
      .collectSync();

    const expected = pl.DataFrame({
      "letters": ["c", "c", "a", "a"],
      "list_1": [1, 2, 1, 3],
    });

    expect(actual).toFrameEqualIgnoringOrder(expected);
  });
  test("fetch", () =>{
    const df = pl.DataFrame({
      "foo": [1, 2],
      "bar": ["a", "b"]
    });
    const expected = pl.DataFrame({
      "foo": [1],
      "bar": ["a"]
    });
    const actual = df
      .lazy()
      .select("*")
      .fetch(1);
    expect(actual).toFrameEqual(expected);
  });
  test("", () =>{});

  test("min", () => {
    const actual = pl.DataFrame({
      "foo": [2, 2, 3, 4],
      "bar": [2, 3, 1, 11]
    }).lazy()
      .min()
      .collectSync();

    const expected = pl.DataFrame({
      "foo": [2],
      "bar": [1]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("max", () => {
    const actual = pl.DataFrame({
      "foo": [2, 2, 3, 4],
      "bar": [2, 3, 1, 11]
    }).lazy()
      .max()
      .collectSync();

    const expected = pl.DataFrame({
      "foo": [4],
      "bar": [11]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("sum", () => {
    const actual = pl.DataFrame({
      "foo": [2, 2, 3, 4],
      "bar": [2, 3, 1, 11]
    }).lazy()
      .sum()
      .collectSync();

    const expected = pl.DataFrame({
      "foo": [11],
      "bar": [17]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("mean", () => {
    const actual = pl.DataFrame({
      "foo": [2, 2, 3, 4],
      "bar": [2, 3, 1, 1]
    }).lazy()
      .mean()
      .collectSync();

    const expected = pl.DataFrame({
      "foo": [2.75],
      "bar": [1.75]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("median", () => {
    const actual = pl.DataFrame({
      "foo": [2, 2, 3, 4],
      "bar": [2, 3, 1, 1]
    }).lazy()
      .median()
      .collectSync();

    const expected = pl.DataFrame({
      "foo": [2.5],
      "bar": [1.5]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("std", () => {
    const actual = pl.DataFrame({
      "foo": [2, 2, 3, 4],
      "bar": [2, 3, 1, 1]
    }).lazy()
      .std()
      .collectSync();

    const expected = pl.DataFrame({
      "foo": [0.9574271077563381],
      "bar": [0.9574271077563381]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("var", () => {
    const actual = pl.DataFrame({
      "foo": [2, 2, 3, 4],
      "bar": [2, 3, 1, 1]
    }).lazy()
      .var()
      .collectSync();

    const expected = pl.DataFrame({
      "foo": [0.9166666666666666],
      "bar": [0.9166666666666666]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("reverse", () => {
    const actual = pl.DataFrame({
      "foo": [2, 2, 3, 4],
      "bar": [2, 3, 1, 1]
    }).lazy()
      .reverse()
      .collectSync();

    const expected = pl.DataFrame({
      "foo": [4, 3, 2, 2],
      "bar": [1, 1, 3, 2]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("tail", () => {
    const actual = pl.DataFrame({
      "foo": [2, 2, 3, 4],
      "bar": [2, 3, 1, 1]
    }).lazy()
      .tail(1)
      .collectSync();

    const expected = pl.DataFrame({
      "foo": [4],
      "bar": [1]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("head", () => {
    const actual = pl.DataFrame({
      "foo": [2, 2, 3, 4],
      "bar": [2, 3, 1, 1]
    }).lazy()
      .head(1)
      .collectSync();

    const expected = pl.DataFrame({
      "foo": [2],
      "bar": [2]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("limit", () => {
    const actual = pl.DataFrame({
      "foo": [2, 2, 3, 4],
      "bar": [2, 3, 1, 1]
    }).lazy()
      .limit(1)
      .collectSync();

    const expected = pl.DataFrame({
      "foo": [2],
      "bar": [2]
    });
    expect(actual).toFrameEqual(expected);
  });
});