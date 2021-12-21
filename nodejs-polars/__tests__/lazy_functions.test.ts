import pl, {col, cols, lit} from "@polars/index";
import {df as _df} from "./setup";

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
  test("arange:positional", () => {
    const df = pl.DataFrame({
      "foo": [1, 1, 1],
    });
    const expected = pl.DataFrame({"foo":  [1, 1]});
    const actual = df.filter(pl.col("foo").gtEq(pl.arange(0, 3)));
    expect(actual).toFrameEqual(expected);
  });
  test("arange:named", () => {
    const df = pl.DataFrame({
      "foo": [1, 1, 1],
    });
    const expected = pl.DataFrame({"foo":  [1, 1]});
    const actual = df.filter(pl.col("foo").gtEq(pl.arange({low: 0, high: 3})));
    expect(actual).toFrameEqual(expected);
  });
  test("arange:eager", () => {
    const df = pl.DataFrame({
      "foo": [1, 1, 1],
    });
    const expected = pl.DataFrame({"foo":  [1, 1]});
    const actual = df.filter(pl.col("foo").gtEq(pl.arange({low: 0, high: 3, eager: true})));
    expect(actual).toFrameEqual(expected);
  });
  test.skip("argSortBy", () => {
    const actual = _df().select(pl.argSortBy(["int_nulls", "floats"], [false, true]))
      .getColumn("int_nulls");
  });
  test("avg", () => {
    const df = pl.DataFrame({"foo": [4, 5, 6, 4, 5, 6]});
    const expected = pl.select(lit(5).as("foo"));
    const actual = df.select(pl.avg("foo"));

    expect(actual).toFrameEqual(expected);
  });
  test("concatList", () => {
    const s0 = pl.Series("a", [[1, 2]]);
    const s1 = pl.Series("b", [[3, 4, 5]]);
    const expected = pl.Series("a", [[1, 2, 3, 4, 5]]);
    const df = pl.DataFrame([s0, s1]);

    const actual = df.select(pl.concatList(["a", "b"]).as("a")).getColumn("a");
    expect(actual).toSeriesEqual(expected);
  });
  test("concatString", () => {
    const s0 = pl.Series("a", ["a", "b", "c"]);
    const s1 = pl.Series("b", ["d", "e", "f"]);
    const expected = pl.Series("concat", ["a,d", "b,e", "c,f"]);
    const df = pl.DataFrame([s0, s1]);
    const actual = df.select(pl.concatString(["a", "b"]).as("concat")).getColumn("concat");
    expect(actual).toSeriesEqual(expected);
  });
  test("concatString:sep", () => {
    const s0 = pl.Series("a", ["a", "b", "c"]);
    const s1 = pl.Series("b", ["d", "e", "f"]);
    const expected = pl.Series("concat", ["a=d", "b=e", "c=f"]);
    const df = pl.DataFrame([s0, s1]);
    const actual = df.select(pl.concatString(["a", "b"], "=").as("concat")).getColumn("concat");
    expect(actual).toSeriesEqual(expected);
  });
  test("concatString:named", () => {
    const s0 = pl.Series("a", ["a", "b", "c"]);
    const s1 = pl.Series("b", ["d", "e", "f"]);
    const expected = pl.Series("concat", ["a=d", "b=e", "c=f"]);
    const df = pl.DataFrame([s0, s1]);
    const actual = df.select(pl.concatString({exprs: ["a", "b"], sep: "="}).as("concat")).getColumn("concat");
    expect(actual).toSeriesEqual(expected);
  });
  test("count:series", () => {
    const s0 = pl.Series([1, 2, 3, 4, 5]);
    const expected = 5;
    const actual = pl.count(s0);
    expect(actual).toStrictEqual(expected);
  });
  test("count:column", () => {
    const s0 = pl.Series("a", [1, 2, 3, 4, 5]).cast(pl.Int32);
    const s1 = pl.Series("b", [11, 22, 33, 44, 55]).cast(pl.Int32);
    const expected = pl.select(
      lit(5)
        .cast(pl.Int32)
        .as("a")
    );
    const actual = pl.DataFrame([s0, s1]).select(pl.count("a"));
    expect(actual).toFrameEqual(expected);
  });
  test("cov", () => {
    const df = pl.DataFrame([
      pl.Series("A", [1, 2, 3, 4, 5]),
      pl.Series("B", [5, 4, 3, 2, 1]),
    ]);
    const actual = df.select(pl.cov("A", "B")).row(0)[0];
    expect(actual).toStrictEqual(-2.5);
  });
  test("cov:expr", () => {
    const df = pl.DataFrame([
      pl.Series("A", [1, 2, 3, 4, 5]),
      pl.Series("B", [5, 4, 3, 2, 1]),
    ]);
    const actual = df.select(pl.cov(col("A"), col("B"))).row(0)[0];
    expect(actual).toStrictEqual(-2.5);
  });
  test("exclude", () => {
    const df = pl.DataFrame([
      pl.Series("A", [1, 2, 3, 4, 5]),
      pl.Series("B", [5, 4, 3, 2, 1]),
      pl.Series("C", ["a", "b", "c", "d", "e"]),
    ]);
    const expected = pl.DataFrame([
      pl.Series("A", [1, 2, 3, 4, 5]),
      pl.Series("C", ["a", "b", "c", "d", "e"]),
    ]);
    const actual = df.select(pl.exclude("B"));
    expect(actual).toFrameEqual(expected);
  });
  test("exclude:multiple", () => {
    const df = pl.DataFrame([
      pl.Series("A", [1, 2, 3, 4, 5]),
      pl.Series("B", [5, 4, 3, 2, 1]),
      pl.Series("C", ["a", "b", "c", "d", "e"]),
    ]);
    const expected = pl.DataFrame([
      pl.Series("C", ["a", "b", "c", "d", "e"]),
    ]);
    const actual = df.select(pl.exclude("A", "B"));
    expect(actual).toFrameEqual(expected);
  });
  test("first:series", () => {
    const s = pl.Series("a", [1, 2, 3]);
    const actual = pl.first(s);
    expect(actual).toStrictEqual(1);
  });
  test("first:df", () => {
    const actual = _df().select(pl.first("bools"))
      .row(0)[0];
    expect(actual).toStrictEqual(false);
  });
  test("first:invalid", () => {
    const s = pl.Series("a", []);
    const fn = () => pl.first(s);
    expect(fn).toThrow();
  });
  test("format:tag", () => {
    const df =  pl.DataFrame({
      "a": ["a", "b", "c"],
      "b": [1, 2, 3],
    });
    const expected = pl.DataFrame({"eq": ["a=a;b=1.0", "a=b;b=2.0", "a=c;b=3.0"]});
    const actual = df.select(pl.format`${lit("a")}=${col("a")};b=${col("b")}`.as("eq"));
    expect(actual).toFrameEqual(expected);
  });
  test("format:pattern", () => {
    const df =  pl.DataFrame({
      "a": ["a", "b", "c"],
      "b": [1, 2, 3],
    });
    const fmt = pl.format("{}={};b={}", lit("a"), col("a"), col("b")).as("eq");
    const expected = pl.DataFrame({"eq": ["a=a;b=1.0", "a=b;b=2.0", "a=c;b=3.0"]});
    const actual = df.select(fmt);
    expect(actual).toFrameEqual(expected);
  });
  test("format:invalid", () => {
    const fn = () => pl.format("{}{}={};b={}", lit("a"), col("a"), col("b")).as("eq");
    expect(fn).toThrow();
  });
  test("head:series", () => {
    const expected = pl.Series("a", [1, 2]);
    const actual = pl.head(pl.Series("a", [1, 2, 3]), 2);
    expect(actual).toSeriesEqual(expected);
  });
  test("head:df", () => {
    const df = pl.DataFrame({
      "a": [1, 2, 5],
      "b": ["foo", "bar", "baz"]
    });
    const expected = pl.DataFrame({
      "a": [1],
      "b": ["foo"]
    });

    const actual = df.select(pl.head("*", 1));
    expect(actual).toFrameEqual(expected);
  });
  test("head:expr", () => {
    const df = pl.DataFrame({
      "a": [1, 2, 5],
      "b": ["foo", "bar", "baz"]
    });
    const expected = pl.DataFrame({
      "a": [1],
      "b": ["foo"]
    });

    const actual = df.select(pl.head(col("*"), 1));
    expect(actual).toFrameEqual(expected);
  });
  test("last:series", () => {
    const actual = pl.last(pl.Series("a", [1, 2, 3]));
    expect(actual).toStrictEqual(3);
  });
  test("last:string", () => {
    const df = pl.DataFrame({
      "a": [1, 2, 5],
      "b": ["foo", "bar", "baz"]
    });
    const actual = df.select(pl.last("b"));
    expect(actual).toFrameEqual(pl.select(lit("baz").as("b")));
  });
  test("last:col", () => {
    const df = pl.DataFrame({
      "a": [1, 2, 5],
      "b": ["foo", "bar", "baz"]
    });
    const actual = df.select(pl.last(col("b")));
    expect(actual).toFrameEqual(pl.select(lit("baz").as("b")));
  });
  test("last:invalid", () => {
    const fn = () => pl.last(pl.Series("a", []));
    expect(fn).toThrow(RangeError);
  });
  test("list", () => {
    const df = pl.DataFrame({
      "a": [1, 2, 3],
      "b": ["a", "b", "c"],
    });
    const expected = pl.DataFrame({
      "a": [1, 2, 3],
      "b": [["a"], ["b"], ["c"]]
    });
    const actual = df.groupBy("a")
      .agg(
        pl.list("b")
          .keepName()
      )
      .sort({by:"a"});
    expect(actual).toFrameEqual(expected);
  });
  test("mean:series", () => {
    const actual = pl.mean(pl.Series([2, 2, 8]));
    expect(actual).toStrictEqual(4);
  });
  test("mean:string", () => {
    const df = pl.DataFrame({"a": [2, 2, 8]});
    const actual = df.select(pl.mean("a")).getColumn("a")[0];
    expect(actual).toStrictEqual(4);
  });
  test("mean:col", () => {
    const df = pl.DataFrame({"a": [2, 2, 8]});
    const actual = df.select(pl.mean(col("a"))).getColumn("a")[0];
    expect(actual).toStrictEqual(4);
  });
  test("median:series", () => {
    const actual = pl.median(pl.Series([2, 2, 8]));
    expect(actual).toStrictEqual(2);
  });
  test("median:string", () => {
    const df = pl.DataFrame({"a": [2, 2, 8]});
    const actual = df.select(pl.median("a")).getColumn("a")[0];
    expect(actual).toStrictEqual(2);
  });
  test("median:col", () => {
    const df = pl.DataFrame({"a": [2, 2, 8]});
    const actual = df.select(pl.median(col("a"))).getColumn("a")[0];
    expect(actual).toStrictEqual(2);
  });
  test("nUnique:series", () => {
    const actual = pl.nUnique(pl.Series([2, 2, 8]));
    expect(actual).toStrictEqual(2);
  });
  test("nUnique:string", () => {
    const df = pl.DataFrame({"a": [2, 2, 8]});
    const actual = df.select(pl.nUnique("a")).getColumn("a")[0];
    expect(actual).toStrictEqual(2);
  });
  test("nUnique:col", () => {
    const df = pl.DataFrame({"a": [2, 2, 8]});
    const actual = df.select(pl.nUnique(col("a"))).getColumn("a")[0];
    expect(actual).toStrictEqual(2);
  });
  test("pearsonCorr", () => {
    const df = pl.DataFrame([
      pl.Series("A", [1, 2, 3, 4]),
      pl.Series("B", [2, 4, 6, 8]),
    ]);
    const actual = df.select(
      pl.pearsonCorr("A", "B").round(1)
    )
      .row(0)[0];
    expect(actual).toStrictEqual(1);
  });
  test("quantile:series", () => {
    const s = pl.Series([1, 2, 3]);
    const actual = pl.quantile(s, 0.5);
    expect(actual).toStrictEqual(2);
  });
  test("quantile:string", () => {
    const df = pl.DataFrame({"a": [1, 2, 3]});
    const actual = df.select(pl.quantile("a", 0.5)).getColumn("a")[0];
    expect(actual).toStrictEqual(2);
  });
  test("quantile:col", () => {
    const df = pl.DataFrame({"a": [1, 2, 3]});
    const actual = df.select(pl.quantile(col("a"), 0.5)).getColumn("a")[0];
    expect(actual).toStrictEqual(2);
  });
  test("spearmanRankCorr", () => {
    const df = pl.DataFrame([
      pl.Series("A", [1, 2, 3, 4]),
      pl.Series("B", [2, 4, 6, 8]),
    ]);
    const actual = df.select(
      pl.spearmanRankCorr("A", "B").round(1)
    )
      .row(0)[0];
    expect(actual).toStrictEqual(1);
  });
  test("tail:series", () => {
    const s = pl.Series("a", [1, 2, 3]);
    const expected = pl.Series("a", [2, 3]);
    const actual = pl.tail(s, 2);
    expect(actual).toSeriesEqual(expected);
  });
  test("tail:string", () => {
    const df = pl.DataFrame({"a": [1, 2, 3]});
    const expected = pl.Series("a", [2, 3]);
    const actual = df.select(pl.tail("a", 2)).getColumn("a");
    expect(actual).toSeriesEqual(expected);
  });
  test("tail:col", () => {
    const df = pl.DataFrame({"a": [1, 2, 3]});
    const expected = pl.Series("a", [2, 3]);
    const actual = df.select(pl.tail(col("a"), 2)).getColumn("a");
    expect(actual).toSeriesEqual(expected);
  });
});
