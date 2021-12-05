/* eslint-disable newline-per-chained-call */
import pl from "@polars";

describe("dataframe", () => {
  const df = pl.DataFrame([
    pl.Series("foo", [1, 2, 9], pl.Int16),
    pl.Series("bar", [6, 2, 8], pl.Int16),
  ]);
  test("emptyDF", () => {
    const df = pl.DataFrame({});
    expect(df.isEmpty()).toStrictEqual(true);
  });
  test("dtypes", () =>{
    const expected = ["Float64", "Utf8"];
    const actual = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).dtypes;
    expect(actual).toEqual(expected);
  });
  test("height", () =>{
    const expected = 3;
    const actual = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).height;
    expect(actual).toEqual(expected);
  });
  test("width", () =>{
    const expected = 2;
    const actual = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).width;
    expect(actual).toEqual(expected);
  });
  test("shape", () =>{
    const expected = {height: 3, width: 2};
    const actual = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).shape;
    expect(actual).toEqual(expected);
  });
  test("columns", () =>{
    const expected = ["a", "b"];
    const actual = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).columns;
    expect(actual).toEqual(expected);
  });
  test("clone", () =>{
    const expected = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]});
    const actual = expected.clone();
    expect(actual).toFrameEqual(expected);
  });
  test.todo("describe");
  test.todo("downsample");
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
    const actual = df.drop("apple");
    expect(actual).toFrameEqual(expected);
  });
  test("dropDuplicates", () =>{
    const actual = pl.DataFrame({
      "foo": [1, 2, 2, 3],
      "bar": [1, 2, 2, 4],
      "ham": ["a", "d", "d", "c"],
    }).dropDuplicates();
    const expected = pl.DataFrame({
      "foo": [1,  2, 3],
      "bar": [1,  2, 4],
      "ham": ["a", "d", "c"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("dropNulls", () =>{
    const actual = pl.DataFrame({
      "foo": [1, null, 2, 3],
      "bar": [6.0, .5, 7.0, 8.0],
      "ham": ["a", "d", "b", "c"],
    }).dropNulls();
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("explode", () => {
    const actual = pl.DataFrame({
      "letters": ["c", "a"],
      "nrs": [[1, 2], [1, 3]]
    }).explode("nrs");

    const expected = pl.DataFrame({
      "letters": ["c", "c", "a", "a"],
      "nrs": [1, 2, 1, 3]
    });

    expect(actual).toFrameEqual(expected);
  });
  test("fillNull:zero", () =>{
    const actual = pl.DataFrame({
      "foo": [1, null, 2, 3],
      "bar": [6.0, .5, 7.0, 8.0],
      "ham": ["a", "d", "b", "c"],
    }).fillNull("zero");
    const expected = pl.DataFrame({
      "foo": [1, 0, 2, 3],
      "bar": [6.0, .5, 7.0, 8.0],
      "ham": ["a", "d", "b", "c"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("fillNull:one", () =>{
    const actual = pl.DataFrame({
      "foo": [1, null, 2, 3],
      "bar": [6.0, .5, 7.0, 8.0],
      "ham": ["a", "d", "b", "c"],
    }).fillNull("one");
    const expected = pl.DataFrame({
      "foo": [1, 1, 2, 3],
      "bar": [6.0, .5, 7.0, 8.0],
      "ham": ["a", "d", "b", "c"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test.todo("filter");
  test("findIdxByName", () => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).findIdxByName("ham");
    const expected = 2;
    expect(actual).toEqual(expected);
  });
  test.todo("fold");
  test("getColumn", () => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).getColumn("ham");
    const expected = pl.Series("ham", ["a", "b", "c"]);
    expect(actual).toSeriesEqual(expected);
  });
  test("getColumns", () => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "ham": ["a", "b", "c"]
    }).getColumns();
    const expected = [
      pl.Series("foo", [1, 2, 3]),
      pl.Series("ham", ["a", "b", "c"])
    ];
    actual.forEach((a, idx) => {
      expect(a).toSeriesEqual(expected[idx]);
    });

  });
  test("groupBy", () => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "ham": ["a", "b", "c"]
    }).groupBy("foo");
    expect(actual.toString()).toEqual("GroupBy");
  });
  test("hashRows", () => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "ham": ["a", "b", "c"]
    }).hashRows();
    expect(actual.dtype).toEqual("UInt64");
  });
  test("head", () => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "ham": ["a", "b", "c"]
    }).head(1);
    const expected  = pl.DataFrame({
      "foo": [1],
      "ham": ["a"]
    }).head(1);
    expect(actual).toFrameEqual(expected);
  });
  test("hstack:series", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).hstack([pl.Series("apple", [10, 20, 30])]);
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"],
      "apple": [10, 20, 30]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("hstack:df", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).hstack(pl.DataFrame([pl.Series("apple", [10, 20, 30])]));
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"],
      "apple": [10, 20, 30]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("hstack:df", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    });
    actual.insertAtIdx(0, pl.Series("apple", [10, 20, 30]));
    const expected = pl.DataFrame({
      "apple": [10, 20, 30],
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test.todo("interpolate");
  test.todo("isDuplicated");
  test.todo("isEmpty");
  test.todo("isUnique");
  test("join", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    const otherDF = pl.DataFrame({
      "apple": ["x", "y", "z"],
      "ham": ["a", "b", "d"]
    });
    const actual = df.join(otherDF, {on: "ham"});

    const expected = pl.DataFrame({
      "foo": [1, 2],
      "bar": [6.0, 7.0],
      "ham": ["a", "b"],
      "apple": ["x", "y"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("lazy", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    }).lazy().collectSync();

    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("limit", () => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "ham": ["a", "b", "c"]
    }).limit(1);
    const expected  = pl.DataFrame({
      "foo": [1],
      "ham": ["a"]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("max:axis:0", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).max();
    expect(actual.row(0)).toEqual([3, 8, null]);
  });
  test("max:axis:1", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 9],
      "bar": [6, 2, 8],
    }).max(1);
    const expected  = pl.Series("foo", [6, 2, 9]);
    expect(actual).toSeriesEqual(expected);
  });
  test("mean:axis:0", () => {
    const actual = pl.DataFrame({
      "foo": [4, 4, 4],
      "bar": [1, 1, 10],
      "ham": ["a", "b", "a"]
    }).mean();
    expect(actual.row(0)).toEqual([4, 4, null]);
  });
  test("mean:axis:1", () => {
    const actual = pl.DataFrame({
      "foo": [1, null, 6],
      "bar": [6, 2, 8],
    }).mean(1, "ignore");
    const expected  = pl.Series("foo", [3.5, 2, 7]);
    expect(actual).toSeriesEqual(expected);
  });
  test("median", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).median();

    expect(actual.row(0)).toEqual([2, 7, null]);
  });
  test.todo("melt");
  test("min:axis:0", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).min();
    expect(actual.row(0)).toEqual([1, 6, null]);
  });
  test("min:axis:1", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 9],
      "bar": [6, 2, 8],
    }).min(1);
    const expected  = pl.Series("foo", [1, 2, 8]);
    expect(actual).toSeriesEqual(expected);
  });
  test("nChunks", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 9],
      "bar": [6, 2, 8],
    }).nChunks();
    expect(actual).toEqual(1);
  });
  test("nullCount", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, null],
      "bar": [6, 2, 8],
      "apple": [6, 2, 8],
      "pizza": [null, null, 8],
    }).nullCount();
    expect(actual.row(0)).toEqual([1, 0, 0, 2]);
  });
  test.todo("pipe");
  test("quantile", ()=>{
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).quantile(0.5);
    expect(actual.row(0)).toEqual([2, 7, null]);
  });
  test("rename", ()=>{
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).rename({
      "foo": "foo_new",
      "bar": "bar_new",
      "ham": "ham_new"
    });
    expect(actual.columns).toEqual(["foo_new", "bar_new", "ham_new"]);
  });
  test("replaceAtIdx", ()=>{
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    });
    const s = pl.Series("new_foo", [0.12, 2.0, 9.99]);
    actual.replaceAtIdx(0, s);
    expect(actual.getColumn("new_foo")).toSeriesEqual(s);
    expect(actual.findIdxByName("new_foo")).toEqual(0);
  });
  test("row", ()=>{
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).row(1);
    expect(actual).toEqual([2, 7, "b"]);
  });
  test("rows", ()=>{
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).rows();
    expect(actual).toEqual([
      [1, 6, "a"],
      [2, 7, "b"],
      [3, 8, "c"]
    ]);
  });
  test("sample:n", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).sample(2);
    expect(actual.height).toStrictEqual(2);
  });
  test("sample:frac", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
      "ham": ["a", "b", "c", null]
    }).sample({frac: 0.5});
    expect(actual.height).toStrictEqual(2);
  });
  test("sample:frac", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
      "ham": ["a", "b", "c", null]
    }).sample({frac: 0.75});
    expect(actual.height).toStrictEqual(3);
  });
  test("select:strings", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
      "ham": ["a", "b", "c", null]
    }).select("ham", "foo");
    const foo = pl.Series("foo", [1, 2, 3, 1]);
    const ham = pl.Series("ham", ["a", "b", "c", null]);
    expect(actual.width).toStrictEqual(2);
    expect(actual.getColumn("foo")).toSeriesEqual(foo);
    expect(actual.getColumn("ham")).toSeriesEqual(ham);
  });
  test("select:expr", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
      "ham": ["a", "b", "c", null]
    }).select(pl.col("foo"), "ham");
    const foo = pl.Series("foo", [1, 2, 3, 1]);
    const ham = pl.Series("ham", ["a", "b", "c", null]);
    expect(actual.width).toStrictEqual(2);
    expect(actual.getColumn("foo")).toSeriesEqual(foo);
    expect(actual.getColumn("ham")).toSeriesEqual(ham);
  });
  test("shift:pos", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).shift(1);
    const expected = pl.DataFrame({
      "foo": [null, 1, 2, 3],
      "bar": [null, 6, 7, 8],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("shift:neg", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).shift(-1);
    const expected = pl.DataFrame({
      "foo": [2, 3, 1, null],
      "bar": [7, 8, 1, null],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("shiftAndFill:positional", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).shiftAndFill(-1, 99);
    const expected = pl.DataFrame({
      "foo": [2, 3, 1, 99],
      "bar": [7, 8, 1, 99],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("shiftAndFill:named", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).shiftAndFill({periods: -1, fillValue: 99});
    const expected = pl.DataFrame({
      "foo": [2, 3, 1, 99],
      "bar": [7, 8, 1, 99],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("shrinkToFit", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).shrinkToFit();
    const expected = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("slice:positional", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).slice(0, 2);
    const expected = pl.DataFrame({
      "foo": [1, 2],
      "bar": [6, 7],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("slice:named", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).slice({offset: 0, length: 2});
    const expected = pl.DataFrame({
      "foo": [1, 2],
      "bar": [6, 7],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("sort:positional", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).sort("bar");
    const expected = pl.DataFrame({
      "foo": [1, 1, 2, 3],
      "bar": [1, 6, 7, 8],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("sort:named", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).sort({by: "bar", reverse: true});
    const expected = pl.DataFrame({
      "foo": [3, 2, 1, 1],
      "bar": [8, 7, 6, 1],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("sort:multi-args", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, -1],
      "bar": [6, 7, 8, 2],
      "baz": ["a", "b", "d", "A"],
    }).sort({
      by: [
        pl.col("baz"),
        pl.col("bar")
      ]
    });
    const expected = pl.DataFrame({
      "foo": [-1, 1, 2, 3],
      "bar": [2, 6, 7, 8],
      "baz": ["A", "a", "b", "d"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("std", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).std();
    const expected = pl.DataFrame([
      pl.Series("foo", [1]),
      pl.Series("bar", [1]),
      pl.Series("ham", [null], pl.Utf8),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("sum:axis:0", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).sum();
    expect(actual.row(0)).toEqual([6, 21, null]);
  });
  test("sum:axis:1", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 9],
      "bar": [6, 2, 8],
    }).sum(1).rename("sum");
    const expected  = pl.Series("sum", [7, 4, 17]);
    expect(actual).toSeriesEqual(expected);
  });
  test("tail", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 9],
      "bar": [6, 2, 8],
    }).tail(1).row(0);
    const expected  = [9, 8];
    expect(actual).toEqual(expected);
  });
  describe("toCSV", () =>{
    test("string", () => {
      const actual = df.clone().toCSV();
      const expected  = "foo,bar\n1,6\n2,2\n9,8\n";
      expect(actual).toEqual(expected);
    });
    test("string:sep", () => {
      const actual = df.clone().toCSV({sep: "X"});
      const expected  = "fooXbar\n1X6\n2X2\n9X8\n";
      expect(actual).toEqual(expected);
    });
    test("string:header", () => {
      const actual = df.clone().toCSV({sep: "X", hasHeader: false});
      const expected  = "1X6\n2X2\n9X8\n";
      expect(actual).toEqual(expected);
    });
  });
  test("toJSON", () =>{
    const rows = [
      {foo: 1.1, bar: 6.2, ham: "a"},
      {foo: 3.1, bar: 9.2, ham: "b"},
      {foo: 3.1, bar: 9.2, ham: "c"}
    ];
    const actual = pl.DataFrame(rows).toJSON();
    const expected = rows.map(r => JSON.stringify(r)).join("\n").concat("\n");
    expect(actual).toEqual(expected);
  });
  test("toSeries", () =>{
    const s = pl.Series([1, 2, 3]);
    const actual = s.clone().toFrame().toSeries(0);
    expect(actual).toSeriesEqual(s);
  });
  test.todo("upsample");
  test("var", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).var();
    const expected = pl.DataFrame([
      pl.Series("foo", [1]),
      pl.Series("bar", [1]),
      pl.Series("ham", [null], pl.Utf8),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("vstack", () => {
    const df1 = pl.DataFrame({
      "foo": [1, 2],
      "bar": [6, 7],
      "ham": ["a", "b"]
    });
    const df2 = pl.DataFrame({
      "foo": [3, 4],
      "bar": [8, 9],
      "ham": ["c", "d"]
    });

    const actual = df1.vstack(df2);
    const expected = pl.DataFrame([
      pl.Series("foo", [1, 2, 3, 4]),
      pl.Series("bar", [6, 7, 8, 9]),
      pl.Series("ham", ["a", "b", "c", "d"]),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("withColumn:series", () => {
    const actual = df
      .clone()
      .withColumn(pl.Series("col_a", ["a", "a", "a"], pl.Utf8));
    const expected =  pl.DataFrame([
      pl.Series("foo", [1, 2, 9], pl.Int16),
      pl.Series("bar", [6, 2, 8], pl.Int16),
      pl.Series("col_a", ["a", "a", "a"], pl.Utf8),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("withColumn:expr", () => {
    const actual = df
      .clone()
      .withColumn(pl.lit("a").alias("col_a"));

    const expected =  pl.DataFrame([
      pl.Series("foo", [1, 2, 9], pl.Int16),
      pl.Series("bar", [6, 2, 8], pl.Int16),
      pl.Series("col_a", ["a", "a", "a"], pl.Utf8),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("withColumns:series", () => {
    const actual = df
      .clone()
      .withColumns(
        pl.Series("col_a", ["a", "a", "a"], pl.Utf8),
        pl.Series("col_b", ["b", "b", "b"], pl.Utf8)
      );
    const expected =  pl.DataFrame([
      pl.Series("foo", [1, 2, 9], pl.Int16),
      pl.Series("bar", [6, 2, 8], pl.Int16),
      pl.Series("col_a", ["a", "a", "a"], pl.Utf8),
      pl.Series("col_b", ["b", "b", "b"], pl.Utf8),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("withColumns:expr", () => {
    const actual = df
      .clone()
      .withColumns(
        pl.lit("a").alias("col_a"),
        pl.lit("b").alias("col_b")
      );
    const expected =  pl.DataFrame([
      pl.Series("foo", [1, 2, 9], pl.Int16),
      pl.Series("bar", [6, 2, 8], pl.Int16),
      pl.Series("col_a", ["a", "a", "a"], pl.Utf8),
      pl.Series("col_b", ["b", "b", "b"], pl.Utf8),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("withColumnRenamed:positional", () => {
    const actual = df
      .clone()
      .withColumnRenamed("foo", "apple");

    const expected =  pl.DataFrame([
      pl.Series("apple", [1, 2, 9], pl.Int16),
      pl.Series("bar", [6, 2, 8], pl.Int16),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("withColumnRenamed:named", () => {
    const actual = df
      .clone()
      .withColumnRenamed({existing: "foo", replacement: "apple"});

    const expected =  pl.DataFrame([
      pl.Series("apple", [1, 2, 9], pl.Int16),
      pl.Series("bar", [6, 2, 8], pl.Int16),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("withRowCount", () => {
    const actual = df
      .clone()
      .withRowCount();

    const expected =  pl.DataFrame([
      pl.Series("row_nr", [0, 1, 2], pl.UInt32),
      pl.Series("foo", [1, 2, 9], pl.Int16),
      pl.Series("bar", [6, 2, 8], pl.Int16),
    ]);
    expect(actual).toFrameEqual(expected);
  });
});
