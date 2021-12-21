import {df} from "./setup";
import pl, {col, lit, quantile} from "@polars/index";

describe("expr", () => {
  test("abs", () => {
    const expected = pl.Series("abs", [1, 2, 3]);
    const actual = pl.select(pl.lit(pl.Series([1, -2, -3])).abs()
      .as("abs")).getColumn("abs");
    expect(actual).toSeriesEqual(expected);
  });
  test.todo("aggGroups");
  test("alias", () => {
    const name = "alias";
    const actual = pl.select(lit("a").alias(name));
    expect(actual.columns[0]).toStrictEqual(name);
  });
  test("and", () => {
    const actual = df()
      .filter(
        col("bools")
          .eq(false)
          .and(col("int").eq(3))
      );
    expect(actual.height).toStrictEqual(1);
  });
  test("argMax", () => {
    const actual = df().select(col("int").argMax())
      .row(0)[0];
    expect(actual).toStrictEqual(2);
  });
  test("argMin", () => {
    const actual = df().select(col("int").argMin())
      .row(0)[0];
    expect(actual).toStrictEqual(0);
  });
  test.each`
  args                | expectedSort 
  ${undefined}        | ${[1, 0, 3, 2]}
  ${true}             | ${[2, 3, 0, 1]}
  ${{reverse: true}}  | ${[2, 3, 0, 1]}
  `("argSort", ({args, expectedSort}) => {
    const df = pl.DataFrame({"a": [1, 0, 2, 1.5]});
    const expected = pl.DataFrame({"argSort": expectedSort});
    const actual = df.select(
      col("a")
        .argSort(args)
        .alias("argSort")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("argUnique", () => {
    const df = pl.DataFrame({"a": [1, 1, 4, 1.5]});
    const expected = pl.DataFrame({"argUnique": [0, 2, 3]});
    const actual = df.select(
      col("a")
        .argUnique()
        .cast(pl.Float64)
        .alias("argUnique")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("as", () => {
    const name = "as";
    const actual = pl.select(lit("a").as(name));
    expect(actual.columns[0]).toStrictEqual(name);
  });
  test("backwardFill", () => {
    const df = pl.DataFrame({"a": [null, 1, null, 3]});
    const expected = pl.DataFrame({"a": [1, 1, 3, 3]});
    const actual = df.select(col("a")
      .backwardFill()
    );
    expect(actual).toFrameEqual(expected);
  });
  test("cast", () => {
    const df = pl.DataFrame({"a": [1, 2, 3]});
    const expected = pl.Series("a", [1, 2, 3], pl.Int16).toFrame();
    const actual = df.select(col("a").cast(pl.Int16));
    expect(actual).toFrameEqual(expected);
  });
  test("cast:strict", () => {
    const df = pl.DataFrame({"a": [11111111111111n, 222222222222222222n, null]});

    const fn = () => df.select(col("a").cast(pl.Int16, true));
    expect(fn).toThrow();
  });
  test("count", () => {
    const df = pl.DataFrame({"a": [1, 0, 3, 4, 6, 0]});
    const expected = pl.DataFrame({"a": [6]});
    const actual = df.select(
      col("a")
        .count()
        .cast(pl.Float64)
    );
    expect(actual).toFrameEqual(expected);
  });
  test.each`
  args                | cumCount 
  ${undefined}        | ${[0, 1, 2]}
  ${true}             | ${[2, 1, 0]}
  ${{reverse: true}}  | ${[2, 1, 0]}
  `("$# cumCount", ({args, cumCount}) => {
    const df = pl.DataFrame({"a": [1, 2, 3]});
    const expected = pl.DataFrame({"a": cumCount});
    const actual = df.select(
      col("a")
        .cumCount(args)
        .cast(pl.Float64)
    );
    expect(actual).toFrameEqual(expected);
  });
  test.each`
  args                | cumMax 
  ${undefined}        | ${[1, 1, 2, 2, 3]}
  ${true}             | ${[3, 3, 3, 3, 3]}
  ${{reverse: true}}  | ${[3, 3, 3, 3, 3]}
  `("$# cumMax", ({args, cumMax}) => {
    const df = pl.DataFrame({"a": [1, 0, 2, 1, 3]});
    const expected = pl.DataFrame({"a": cumMax});
    const actual = df.select(
      col("a")
        .cumMax(args)
        .cast(pl.Float64)
    );
    expect(actual).toFrameEqual(expected);
  });
  test.each`
  args                | cumMin 
  ${undefined}        | ${[1, 0, 0, 0, 0]}
  ${true}             | ${[0, 0, 1, 1, 3]}
  ${{reverse: true}}  | ${[0, 0, 1, 1, 3]}
  `("$# cumMin", ({args, cumMin}) => {
    const df = pl.DataFrame({"a": [1, 0, 2, 1, 3]});
    const expected = pl.DataFrame({"a": cumMin});
    const actual = df.select(
      col("a")
        .cumMin(args)
        .cast(pl.Float64)
    );
    expect(actual).toFrameEqual(expected);
  });
  test.each`
  args                | cumProd 
  ${undefined}        | ${[1, 2, 4, 4, 12]}
  ${true}             | ${[12, 12, 6, 3, 3]}
  ${{reverse: true}}  | ${[12, 12, 6, 3, 3]}
  `("$# cumProd", ({args, cumProd}) => {
    const df = pl.DataFrame({"a": [1, 2, 2, 1, 3]});
    const expected = pl.DataFrame({"a": cumProd});
    const actual = df.select(
      col("a")
        .cast(pl.Int64)
        .cumProd(args)
        .cast(pl.Float64)
    );
    expect(actual).toFrameEqual(expected);
  });
  test.each`
  args                | cumSum 
  ${undefined}        | ${[1, 3, 5, 6, 9]}
  ${true}             | ${[9, 8, 6, 4, 3]}
  ${{reverse: true}}  | ${[9, 8, 6, 4, 3]}
  `("cumSum", ({args, cumSum}) => {
    const df = pl.DataFrame({"a": [1, 2, 2, 1, 3]});
    const expected = pl.DataFrame({"a": cumSum});
    const actual = df.select(
      col("a")
        .cumSum(args)
        .cast(pl.Float64)
    );
    expect(actual).toFrameEqual(expected);
  });
  test.each`
  args                                | diff
  ${[1, "ignore"]}                    | ${[null, 2, null, null, 7]}
  ${[{n: 1, nullBehavior: "ignore"}]} | ${[null, 2, null, null, 7]}
  ${[{n: 1, nullBehavior: "drop"}]}   | ${[2, null, null, 7]}
  ${[1, "drop"]}                      | ${[2, null, null, 7]}
  ${[2, "drop"]}                      | ${[null, 0, null]}
  ${[{n: 2, nullBehavior: "drop"}]}   | ${[null, 0, null]}
  `("$# diff", ({args, diff}: any) => {
    const df = pl.DataFrame({"a": [1, 3, null, 3, 10]});
    const expected = pl.DataFrame({"a": diff});
    const actual = df.select(
      (col("a").diff as any)(...args)
    );
    expect(actual).toFrameEqual(expected);

  });
  test("dot", () => {
    const df = pl.DataFrame({
      "a": [1, 2, 3, 4],
      "b": [2, 2, 2, 2]
    });
    const expected = pl.DataFrame({"a": [20]});

    const actual0 = df.select(col("a").dot(col("b")));
    const actual1 = df.select(col("a").dot("b"));
    expect(actual0).toFrameEqual(expected);
    expect(actual1).toFrameEqual(expected);

  });
  test.each`
  other                | eq
  ${col("b")}          | ${[false, true, false]}
  ${lit(1)}            | ${[true, false, false]}
  `("$# eq", ({other, eq}) => {
    const df = pl.DataFrame({
      "a": [1, 2, 3],
      "b": [2, 2, 2]
    });
    const expected = pl.DataFrame({"eq": eq});
    const actual = df.select(
      col("a")
        .eq(other)
        .as("eq")
    );
    expect(actual).toFrameEqual(expected);

  });
  test("exclude", () => {
    const df = pl.DataFrame({
      "a": [1, 2, 3],
      "b": [2, 2, 2],
      "c": ["mac", "windows", "linux"]
    });
    const expected = pl.DataFrame({"a": [1, 2, 3]});
    const actual = df.select(col("*").exclude("b", "c"));
    expect(actual).toFrameEqual(expected);
  });
  test("explode", () => {
    const df = pl.DataFrame({
      "letters": ["c", "a"],
      "nrs": [[1, 2], [1, 3]]
    });
    const expected = pl.DataFrame({
      "nrs": [1, 2, 1, 3]
    });
    const actual = df.select(col("nrs").explode());
    expect(actual).toFrameEqual(expected);

  });
  test.each`
  replacement | filled
  ${lit(1)} | ${1}
  ${2}      | ${2}
  `("$# fillNan", ({replacement, filled}) => {
    const df = pl.DataFrame({
      "a": [1, NaN, 2],
      "b": [2, 1, 1]
    });
    const expected = pl.DataFrame({"fillNan": [1, filled, 2]});
    const actual = df.select(
      col("a")
        .fillNan(replacement)
        .as("fillNan")
    );
    expect(actual).toFrameEqual(expected);
  });
  test.each`
  fillValue | filled
  ${"backward"} | ${[1, 2, 2, 9, 9, null]}
  ${"forward"}  | ${[1, 1, 2, 2, 9, 9]}
  ${"mean"}     | ${[1, 4, 2, 4, 9, 4]}
  ${"min"}      | ${[1, 1, 2, 1, 9, 1]}
  ${"max"}      | ${[1, 9, 2, 9, 9, 9]}
  ${"zero"}     | ${[1, 0, 2, 0, 9, 0]}
  ${"one"}      | ${[1, 1, 2, 1, 9, 1]}
  ${-1}         | ${[1, -1, 2, -1, 9, -1]}
  `("$# fillNull:'$fillValue'", ({fillValue, filled}) => {
    const df = pl.DataFrame({"a": [1, null, 2, null, 9, null]});
    const expected = pl.DataFrame({"fillNull": filled});
    const actual = df.select(
      col("a")
        .fillNull(fillValue)
        .as("fillNull")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("filter", () => {
    const df = pl.DataFrame({"a": [-1, 2, -3, 4]});
    const expected = pl.DataFrame({"a": [2, 4]});
    const actual = df.select(col("a").filter(col("a").gt(0)));
    expect(actual).toFrameEqual(expected);
  });
  test("first", () => {
    const df = pl.DataFrame({"a": [0, 1, 2]});
    const expected = pl.DataFrame({"a": [0]});
    const actual = df.select(col("a").first());
    expect(actual).toFrameEqual(expected);

  });
  test("flatten", () => {
    const df = pl.DataFrame({
      "letters": ["c", "a"],
      "nrs": [[1, 2], [1, 3]]
    });
    const expected = pl.DataFrame({
      "nrs": [1, 2, 1, 3]
    });
    const actual = df.select(col("nrs").flatten());
    expect(actual).toFrameEqual(expected);
  });
  test("floor", () => {
    const df = pl.DataFrame({"a": [0.2, 1, 2.9]});
    const expected = pl.DataFrame({"a": [0, 1, 2]});
    const actual = df.select(col("a").floor());
    expect(actual).toFrameEqual(expected);
  });
  test("forwardFill", () => {
    const df = pl.DataFrame({"a": [1, null, 2, null, 9, null]});
    const expected = pl.DataFrame({"forwardFill": [1, 1, 2, 2, 9, 9]});
    const actual = df.select(
      col("a")
        .forwardFill()
        .as("forwardFill")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("gt", () => {
    const df = pl.DataFrame({"a": [0, 1, 2, -1]});
    const expected = pl.DataFrame({"a": [false, true, true, false]});
    const actual = df.select(col("a").gt(0));
    expect(actual).toFrameEqual(expected);
  });
  test("gtEq", () => {
    const df = pl.DataFrame({"a": [0, 1, 2, -1]});
    const expected = pl.DataFrame({"a": [true, true, true, false]});
    const actual = df.select(col("a").gtEq(0));
    expect(actual).toFrameEqual(expected);
  });
  test.each`
  args       | hashValue
  ${[0]}     | ${6340063056640878722n}
  ${[1n, 1]} | ${9788354747012366704n}
  `("$# hash", ({args, hashValue}) => {
    const df = pl.DataFrame({"a": [1]});
    const expected = pl.DataFrame({"hash": [hashValue]});
    const actual = df.select(
      (col("a").hash as any)(...args)
        .as("hash")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("head", () => {
    const df = pl.DataFrame({"a": [0, 1, 2]});
    const expected = pl.DataFrame({"a": [0, 1]});
    const actual0 = df.select(col("a").head(2));
    const actual1 = df.select(col("a").head({length:2}));
    expect(actual0).toFrameEqual(expected);
    expect(actual1).toFrameEqual(expected);
  });
  test("interpolate", () => {
    const df = pl.DataFrame({"a": [0, null, 2]});
    const expected = pl.DataFrame({"a": [0, 1, 2]});
    const actual = df.select(col("a").interpolate());
    expect(actual).toFrameEqual(expected);

  });
  test("isDuplicated", () => {
    const df = pl.DataFrame({"a": [0, 1, 2, 2]});
    const expected = pl.DataFrame({"a": [false, false, true, true]});
    const actual = df.select(col("a").isDuplicated());
    expect(actual).toFrameEqual(expected);
  });
  test("isFinite", () => {
    const df = pl.DataFrame({"a": [0, Number.POSITIVE_INFINITY, 1]});
    const expected = pl.DataFrame({"a": [true, false, true]});
    const actual = df.select(col("a").isFinite());
    expect(actual).toFrameEqual(expected);
  });
  test("isFirst", () => {
    const df = pl.DataFrame({"a": [0, 1, 2, 2]});
    const expected = pl.DataFrame({"a": [true, true, true, false]});
    const actual = df.select(col("a").isFirst());
    expect(actual).toFrameEqual(expected);
  });
  test("isIn:list", () => {
    const df = pl.DataFrame({"a": [1, 2, 3]});
    const expected = pl.DataFrame({"a": [true, true, false]});
    const actual = df.select(col("a").isIn([1, 2]));
    expect(actual).toFrameEqual(expected);
  });
  test("isIn:expr-eval", () => {
    const df = pl.DataFrame({
      "sets": [[1, 2, 3], [1, 2], [9, 10]],
      "optional_members": [1, 2, 3]
    });
    const expected = pl.DataFrame({"isIn": [true, true, false]});
    const actual = df.select(
      col("optional_members")
        .isIn("sets")
        .as("isIn")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("isIn:expr", () => {
    const df = pl.DataFrame({
      "sets": [[1, 2, 3], [1, 2], [9, 10]],
      "optional_members": [1, 2, 3]
    });
    const expected = pl.DataFrame({"isIn": [true, true, false]});
    const actual = df.select(
      col("optional_members")
        .isIn(col("sets"))
        .as("isIn")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("isInfinite", () => {
    const df = pl.DataFrame({"a": [0, Number.POSITIVE_INFINITY, 1]});
    const expected = pl.DataFrame({"a": [false, true, false]});
    const actual = df.select(col("a").isInfinite());
    expect(actual).toFrameEqual(expected);
  });
  test("isNan", () => {
    const df = pl.DataFrame({"a": [1, NaN, 2]});
    const expected = pl.DataFrame({"isNan": [false, true, false]});
    const actual = df.select(
      col("a")
        .isNan()
        .as("isNan")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("isNotNan", () => {
    const df = pl.DataFrame({"a": [1, NaN, 2]});
    const expected = pl.DataFrame({"isNotNan": [true, false, true]});
    const actual = df.select(
      col("a")
        .isNotNan()
        .as("isNotNan")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("isNull", () => {
    const df = pl.DataFrame({"a": [1, null, 2]});
    const expected = pl.DataFrame({"isNull": [false, true, false]});
    const actual = df.select(
      col("a")
        .isNull()
        .as("isNull")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("isNotNull", () => {
    const df = pl.DataFrame({"a": [1, null, 2]});
    const expected = pl.DataFrame({"isNotNull": [true, false, true]});
    const actual = df.select(
      col("a")
        .isNotNull()
        .as("isNotNull")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("isUnique", () => {
    const df = pl.DataFrame({"a": [1, 1, 2]});
    const expected = pl.DataFrame({"isUnique": [false, false, true]});
    const actual = df.select(
      col("a")
        .isUnique()
        .as("isUnique")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("keepName", () => {
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
        col("b")
          .list()
          .keepName()
      )
      .sort({by:"a"});
    expect(actual).toFrameEqual(expected);
  });
  test("kurtosis", () => {
    const df = pl.DataFrame({"a": [1, 2, 3, 3, 4]});
    const expected = pl.DataFrame({
      "kurtosis": [-1.044]
    });
    const actual = df.select(
      col("a")
        .kurtosis()
        .round({decimals: 3})
        .as("kurtosis")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("last", () => {
    const df = pl.DataFrame({"a": [1, 2, 3]});
    const expected = pl.DataFrame({"last": [3]});
    const actual = df.select(
      col("a")
        .last()
        .as("last")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("list", () => {
    const df = pl.DataFrame({
      "a": ["a", "b", "c"],
    });
    const expected = pl.DataFrame({
      "list": [["a", "b", "c"]]
    });
    const actual = df.select(
      col("a")
        .list()
        .alias("list")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("lowerBound", () => {
    const df = pl.DataFrame([
      pl.Series("int16", [1, 2, 3], pl.Int16),
      pl.Series("int32", [1, 2, 3], pl.Int32),
      pl.Series("int64", [1n, 2n, 3n], pl.Int64),
      pl.Series("uint16", [1, 2, 3], pl.UInt16),
      pl.Series("uint32", [1, 2, 3], pl.UInt32),
      pl.Series("uint64", [1n, 2n, 3n], pl.UInt64),
    ]);

    const expected = pl.DataFrame([
      pl.Series("int16", [-32768], pl.Int16),
      pl.Series("int32", [-2147483648], pl.Int32),
      pl.Series("int64", [-9223372036854775808n], pl.Int64),
      pl.Series("uint16", [0], pl.UInt16),
      pl.Series("uint32", [0], pl.UInt32),
      pl.Series("uint64", [0n], pl.UInt64),
    ]);

    const actual = df.select(
      col("*")
        .lowerBound()
        .keepName()
    );
    expect(actual).toFrameStrictEqual(expected);
  });
  test("lt", () => {
    const df = pl.DataFrame({"a": [1, 2, 3]});
    const expected = pl.DataFrame({"lt": [true, false, false]});
    const actual = df.select(
      col("a")
        .lt(2)
        .as("lt")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("ltEq", () => {
    const df = pl.DataFrame({"a": [1, 2, 3]});
    const expected = pl.DataFrame({"lt": [true, true, false]});
    const actual = df.select(
      col("a")
        .ltEq(2)
        .as("lt")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("max", () => {
    const df = pl.DataFrame({"a": [1, 5, 3]});
    const expected = pl.DataFrame({"max": [5]});
    const actual = df.select(
      col("a")
        .max()
        .as("max")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("mean", () => {
    const df = pl.DataFrame({"a": [2, 2, 8]});
    const expected = pl.DataFrame({"mean": [4]});
    const actual = df.select(
      col("a")
        .mean()
        .as("mean")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("median", () => {
    const df = pl.DataFrame({"a": [6, 1, 2]});
    const expected = pl.DataFrame({"median": [2]});
    const actual = df.select(
      col("a")
        .median()
        .as("median")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("min", () => {
    const df = pl.DataFrame({"a": [2, 3, 1, 2, 1]});
    const expected = pl.DataFrame({"min": [1]});
    const actual = df.select(
      col("a")
        .min()
        .as("min")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("mode", () => {
    const df = pl.DataFrame({"a": [2, 2, 1, 3, 4, 1, 2]});
    const expected = pl.DataFrame({"mode": [2]});
    const actual = df.select(
      col("a")
        .cast(pl.Int64)
        .mode()
        .cast(pl.Float64)
        .as("mode")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("neq", () => {
    const df = pl.DataFrame({"a": [0, 1, 2]});
    const expected = pl.DataFrame({"neq": [true, false, true]});
    const actual = df.select(
      col("a")
        .neq(1)
        .as("neq")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("not", () => {
    const df = pl.DataFrame({"a": [true, true, false]});
    const expected = pl.DataFrame({"not": [false, false, true]});
    const actual = df.select(
      col("a")
        .not()
        .as("not")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("nUnique", () => {
    const df = pl.DataFrame({"a": [0, 1, 2]});
    const expected = pl.DataFrame({"nUnique": [3]});
    const actual = df.select(
      col("a")
        .nUnique()
        .as("nUnique")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("or", () => {
    const df = pl.DataFrame({
      "a": [0, null, 1],
      "b": [0, 1, 2]
    });
    const expected = pl.DataFrame({
      "a": [null, 1],
      "b": [1, 2]
    });
    const actual = df.where(
      col("a")
        .isNull()
        .or(col("b").eq(2))
    );
    expect(actual).toFrameEqual(expected);
  });
  test("over", () => {
    const df = pl.DataFrame({
      "groups": [1, 1, 2, 2],
      "values": [1, 2, 3, 4],
    });
    const expected = pl.DataFrame({
      "groups": [1, 1, 2, 2],
      "sum_groups": [2, 2, 4, 4]
    });
    const actual = df.select(
      col("groups"),
      col("groups")
        .sum()
        .over("groups")
        .alias("sum_groups")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("pow", () => {
    const df = pl.DataFrame({"a": [2, 5, 10]});
    const expected = pl.DataFrame({"pow": [4, 25, 100]});
    const actual = df.select(
      col("a")
        .pow(2)
        .as("pow")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("pow:named", () => {
    const df = pl.DataFrame({"a": [2, 5, 10]});
    const expected = pl.DataFrame({"pow": [4, 25, 100]});
    const actual = df.select(
      col("a")
        .pow({exponent: 2})
        .as("pow")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("prefix", () => {
    const df = pl.DataFrame({"a": [2, 5, 10]});
    const expected = pl.DataFrame({"prefixed_a": [2, 5, 10]});
    const actual = df.select(col("a").prefix("prefixed_"));
    expect(actual).toFrameEqual(expected);
  });
  test("quantile", () => {
    const df = pl.DataFrame({"a": [1, 2, 3]});
    const expected = pl.DataFrame({"quantile": [2]});
    const actual = df.select(
      col("a")
        .quantile(0.5)
        .as("quantile")
    );
    expect(actual).toFrameEqual(expected);
  });

  test.each`
  rankMethod             | ranking
  ${"average"}           | ${[2, 4, 6.5, 4, 4, 6.5, 1]} 
  ${"min"}               | ${[2, 3, 6, 3, 3, 6, 1]} 
  ${"max"}               | ${[2, 5, 7, 5, 5, 7, 1]} 
  ${"dense"}             | ${[2, 3, 4, 3, 3, 4, 1]} 
  ${"ordinal"}           | ${[2, 3, 6, 4, 5, 7, 1]} 
  ${{method: "average"}} | ${[2, 4, 6.5, 4, 4, 6.5, 1]} 
  ${{method: "min"}}     | ${[2, 3, 6, 3, 3, 6, 1]} 
  ${{method: "max"}}     | ${[2, 5, 7, 5, 5, 7, 1]} 
  ${{method: "dense"}}   | ${[2, 3, 4, 3, 3, 4, 1]} 
  ${{method: "ordinal"}} | ${[2, 3, 6, 4, 5, 7, 1]} 
  `("rank: $rankMethod", ({rankMethod, ranking}) => {
    const df = pl.DataFrame({"a": [1, 2, 3, 2, 2, 3, 0]});
    const expected = pl.DataFrame({"rank": ranking});
    const actual = df.select(
      col("a")
        .rank(rankMethod)
        .alias("rank")
    );
    expect(actual).toFrameEqual(expected);
  });

  test("reinterpret", () => {
    const df = pl.DataFrame([
      pl.Series("a", [1n, 2n, 3n], pl.UInt64)
    ]);
    const expected = pl.Series("a", [1n, 2n, 3n], pl.Int64);
    const actual = df.select(col("a").reinterpret()).getColumn("a");
    expect(actual).toSeriesStrictEqual(expected);
  });
  test("repeatBy", () => {
    const df = pl.DataFrame({
      "a": ["a", "b", "c"],
      "n": [1, 2, 1]
    });
    const expected = pl.DataFrame({
      "repeated": [["a"], ["b", "b"], ["c"]]
    });
    const actual = df.select(
      col("a")
        .repeatBy("n")
        .as("repeated")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("reverse", () => {
    const df = pl.DataFrame({"a": ["a", "b", "c"]});
    const expected = pl.DataFrame({
      "a": ["a", "b", "c"],
      "reversed_a": ["c", "b", "a"]
    });
    const actual = df.withColumn(
      col("a")
        .reverse()
        .prefix("reversed_")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("rollingMedian", () => {
    const df = pl.DataFrame({"a": [1, 2, 3, 3, 2, 10, 8]});
    const expected = pl.DataFrame({
      "a": [1, 2, 3, 3, 2, 10, 8],
      "rolling_median_a": [null, 1.5, 2.5, 3, 2.5, 6, 9]
    });
    const actual = df.withColumn(
      col("a")
        .rollingMedian({windowSize: 2})
        .prefix("rolling_median_")
    );
    expect(actual).toFrameStrictEqual(expected);
  });
  test("rollingQuantile", () => {
    const df = pl.DataFrame({"a": [1, 2, 3, 3, 2, 10, 8]});
    const expected = pl.DataFrame({
      "a": [1, 2, 3, 3, 2, 10, 8],
      "rolling_quantile_a": [null, 2, 3, 3, 3, 10, 10]
    });
    const actual = df.withColumn(
      col("a")
        .rollingQuantile({windowSize: 2, quantile: 0.5})
        .prefix("rolling_quantile_")
    );
    expect(actual).toFrameStrictEqual(expected);
  });
  test("rollingSkew", () => {
    const df = pl.DataFrame({"a": [1, 2, 3, 3, 2, 10, 8]});
    const expected = pl.DataFrame({
      "a": [1, 2, 3, 3, 2, 10, 8],
      "bias_true": [null, null, null, "-0.49338220021815865", "0.0", "1.097025449363867", "0.09770939201338157"],
      "bias_false": [null, null, null, "-0.8545630383279711", "0.0", "1.9001038154942962", "0.16923763134384154"]
    });
    const actual = df.withColumns(
      col("a")
        .cast(pl.UInt64)
        .rollingSkew({windowSize: 4})
        .cast(pl.Utf8) // casted to string to retain precision when extracting to JS
        .as("bias_true"),
      col("a")
        .cast(pl.UInt64)
        .rollingSkew({windowSize: 4, bias:false})
        .cast(pl.Utf8) // casted to string to retain precision when extracting to JS
        .as("bias_false")
    );
    expect(actual).toFrameStrictEqual(expected);
  });
  test("round", () => {
    const df = pl.DataFrame({"a": [1.00123, 2.32878, 3.3349999]});
    const expected = pl.DataFrame({"rounded": [1, 2.33, 3.33]});
    const actual  = df.select(
      col("a")
        .round({decimals: 2})
        .as("rounded")
    );
    expect(actual).toFrameStrictEqual(expected);
  });
  test("shift", () => {
    const df = pl.DataFrame({"a": [1, 2, 3, 4]});
    const expected = pl.DataFrame({
      "a": [1, 2, 3, 4],
      "one": [null, 1, 2, 3],
      "negative_one": [2, 3, 4, null],
      "two": [null, null, 1, 2],
    });
    const shifts = pl.DataFrame({
      "name": ["one", "negative_one", "two"],
      "values": [1, -1, 2]
    })
      .map(([name, value]) => col("a")
        .shift(value)
        .as(name)
      );
    const actual  = df.select(
      col("a"),
      ...shifts
    );
    expect(actual).toFrameStrictEqual(expected);
  });
  test("shiftAndFill", () => {
    const df = pl.DataFrame({"a": [1, 2, 3, 4]});
    const expected = pl.DataFrame({
      "a": [1, 2, 3, 4],
      "one": [0, 1, 2, 3],
      "negative_one": [2, 3, 4, 99],
      "two": [2, 2, 1, 2],
    });
    const shifts = [
      ["one", 1, 0],
      ["negative_one", -1, 99],
      ["two", 2, 2]
    ]
      .map(([name, periods, fillValue]: any[]) => {
        return col("a")
          .shiftAndFill({periods, fillValue})
          .as(name);
      });
    const actual  = df.select(
      col("a"),
      ...shifts
    );
    expect(actual).toFrameStrictEqual(expected);
  });
  test("skew", () => {
    const df = pl.DataFrame({"a": [1, 2, 3, 3]});
    const expected = pl.DataFrame({
      "skew:bias=true": ["-0.49338220021815865"],
      "skew:bias=false": ["-0.8545630383279711"]
    });
    const actual = df.select(
      col("a")
        .cast(pl.UInt64)
        .skew()
        .cast(pl.Utf8) // casted to string to retain precision when extracting to JS
        .as("skew:bias=true"),
      col("a")
        .cast(pl.UInt64)
        .skew({bias:false})
        .cast(pl.Utf8) // casted to string to retain precision when extracting to JS
        .as("skew:bias=false")
    );
    expect(actual).toFrameStrictEqual(expected);
  });
  test("slice", () => {
    const df = pl.DataFrame({"a": [1, 2, 3, 4]});
    const expected = pl.DataFrame({
      "slice(0,2)": [1, 2],
      "slice(-2,2)": [3, 4],
      "slice(1,2)": [2, 3],
    });
    const actual = df.select(
      col("a")
        .slice(0, 2)
        .as("slice(0,2)"),
      col("a")
        .slice(-2, 2)
        .as("slice(-2,2)"),
      col("a")
        .slice({offset:1, length:2})
        .as("slice(1,2)"),
    );
    expect(actual).toFrameEqual(expected);
  });
  test("sort", () => {
    const df = pl.DataFrame({
      "a": [1, null, 2, 0],
      "b": [null, "a", "b", "a"]
    });
    const expected = pl.DataFrame({
      "a_sorted_default": [null, 0, 1, 2],
      "a_sorted_reverse": [null, 2, 1, 0],
      "a_sorted_nulls_last": [0, 1, 2, null],
      "a_sorted_reverse_nulls_last": [2, 1, 0, null],
      "b_sorted_default": [null, "a", "a", "b"],
      "b_sorted_reverse": [null, "b", "a", "a"],
      "b_sorted_nulls_last": ["a", "a", "b", null],
      "b_sorted_reverse_nulls_last": ["b", "a", "a", null],
    });
    const a = col("a");
    const b = col("b");
    const actual = df.select(
      a.sort().as("a_sorted_default"),
      a.sort({reverse:true}).as("a_sorted_reverse"),
      a.sort({nullsLast:true}).as("a_sorted_nulls_last"),
      a.sort({reverse:true, nullsLast:true}).as("a_sorted_reverse_nulls_last"),
      b.sort().as("b_sorted_default"),
      b.sort({reverse:true}).as("b_sorted_reverse"),
      b.sort({nullsLast:true}).as("b_sorted_nulls_last"),
      b.sort({reverse:true, nullsLast:true}).as("b_sorted_reverse_nulls_last")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("std", () => {
    const df = pl.DataFrame({"a": [1, 2, 3, 10, 200]});
    const expected = pl.DataFrame({"std": ["87.73"]});
    const actual = df.select(
      col("a")
        .std()
        .round({decimals: 2})
        .cast(pl.Utf8)
        .as("std")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("suffix", () => {
    const df = pl.DataFrame({"a": [2, 5, 10]});
    const expected = pl.DataFrame({"a_suffixed": [2, 5, 10]});
    const actual = df.select(col("a").suffix("_suffixed"));
    expect(actual).toFrameEqual(expected);
  });
  test("sum", () => {
    const df = pl.DataFrame({"a": [2, 5, 10]});
    const expected = pl.DataFrame({"sum": [17]});
    const actual = df.select(
      col("a")
        .sum()
        .as("sum")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("tail", () => {
    const df = pl.DataFrame({"a": [1, 2, 2, 3, 3, 8, null, 1]});
    const expected = pl.DataFrame({
      "tail3": [8, null, 1],
    });
    const actual = df.select(
      col("a")
        .tail(3)
        .as("tail3")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("take", () => {
    const df = pl.DataFrame({"a": [1, 2, 2, 3, 3, 8, null, 1]});
    const expected = pl.DataFrame({
      "take:array": [1, 2, 3, 8],
      "take:list": [1, 2, 2, 3]
    });
    const actual = df.select(
      col("a")
        .take([0, 2, 3, 5])
        .as("take:array"),
      col("a")
        .take(lit([0, 1, 2, 3]))
        .as("take:list")
    );
    expect(actual).toFrameEqual(expected);
  });
  test("takeEvery", () => {
    const df = pl.DataFrame({"a": [1, 1, 2, 2, 3, 3, 8, null, 1]});
    const expected = pl.DataFrame({
      "everyother": [1, 2, 3, 8, 1],
    });
    const actual = df.select(
      col("a")
        .takeEvery(2)
        .as("everyother"),
    );
    expect(actual).toFrameEqual(expected);
  });
  test("unique", () => {
    const df = pl.DataFrame({"a": [1, 1, 2, 2, 3, 3, 8, null, 1]});
    const expected = pl.DataFrame({
      "uniques": [1, 2, 3, 8, null],
    });
    const actual = df.select(
      col("a")
        .unique()
        .sort({nullsLast:true})
        .as("uniques"),
    );
    expect(actual).toFrameEqual(expected);
  });
  test("upperBound", () => {
    const df = pl.DataFrame([
      pl.Series("int16", [1, 2, 3], pl.Int16),
      pl.Series("int32", [1, 2, 3], pl.Int32),
      pl.Series("int64", [1n, 2n, 3n], pl.Int64),
      pl.Series("uint16", [1, 2, 3], pl.UInt16),
      pl.Series("uint32", [1, 2, 3], pl.UInt32),
      pl.Series("uint64", [1n, 2n, 3n], pl.UInt64),
    ]);
    const expected = pl.DataFrame([
      pl.Series("int16", [32767],  pl.Int16),
      pl.Series("int32", [2147483647],  pl.Int32),
      pl.Series("int64", [9223372036854775807n],  pl.Int64),
      pl.Series("uint16", [65535],  pl.UInt16),
      pl.Series("uint32", [4294967295],  pl.UInt32),
      pl.Series("uint64", [18446744073709551615n], pl.UInt64),
    ]);
    const actual = df.select(
      col("*")
        .upperBound()
        .keepName()
    );
    expect(actual).toFrameStrictEqual(expected);
  });
  test("var", () => {
    const df = pl.DataFrame({"a": [1, 1, 2, 3, 8, null]});
    const expected = pl.DataFrame({
      "var": [8.5],
    });
    const actual = df.select(
      col("a")
        .var()
        .as("var"),
    );
    expect(actual).toFrameEqual(expected);
  });
  test("where", () => {
    const df = pl.DataFrame({"a": [-1, 2, -3, 4]});
    const expected = pl.DataFrame({"a": [2, 4]});
    const actual = df.select(col("a").where(col("a").gt(0)));
    expect(actual).toFrameEqual(expected);
  });
});
describe("expr.str", () => {
  test("concat", () => {
    const df = pl.DataFrame({"os": ["kali", "debian", "ubuntu"]});
    const expected = "kali,debian,ubuntu";
    const actualFromSeries = df.getColumn("os").str.concat(",")[0];

    const actual = df.select(col("os").str.concat(",")).row(0)[0];
    expect(actual).toStrictEqual(expected);
    expect(actualFromSeries).toStrictEqual(expected);
  });
  test("contains", () => {
    const df = pl.DataFrame({"os": ["linux-kali", "linux-debian", "windows-vista"]});
    const expected = pl.DataFrame({
      "os": ["linux-kali", "linux-debian", "windows-vista"],
      "isLinux": [true, true, false]
    });
    const seriesActual = df.getColumn("os")
      .str
      .contains("linux")
      .rename("isLinux");
    const actual = df.withColumn(
      col("os")
        .str
        .contains("linux")
        .as("isLinux")
    );
    expect(actual).toFrameEqual(expected);
    expect(seriesActual).toSeriesEqual(expected.getColumn("isLinux"));
  });
  test("extract", () => {
    const df = pl.DataFrame({
      "a": [
        "http://vote.com/ballon_dor?candidate=messi&ref=polars",
        "http://vote.com/ballon_dor?candidat=jorginho&ref=polars",
        "http://vote.com/ballon_dor?candidate=ronaldo&ref=polars"
      ]
    });
    const expected = pl.DataFrame({
      "candidate": ["messi", null, "ronaldo"]
    });

    const seriesActual = df.getColumn("a")
      .str
      .extract(/candidate=(\w+)/g, 1)
      .rename("candidate")
      .toFrame();

    const actual = df.select(
      col("a")
        .str
        .extract(/candidate=(\w+)/g, 1)
        .as("candidate")
    );
    expect(actual).toFrameEqual(expected);
    expect(seriesActual).toFrameEqual(expected);
  });
  test("jsonPathMatch", () => {
    const df = pl.DataFrame({
      "data": [
        `{"os": "linux",   "arch": "debian"}`,
        `{"os": "mac",     "arch": "sierra"}`,
        `{"os": "windows", "arch": "11"}`,
      ]
    });
    const expected = pl.DataFrame({
      "os": ["linux", "mac", "windows"]
    });
    const seriesActual = df.getColumn("data")
      .str
      .jsonPathMatch("$.os")
      .rename("os")
      .toFrame();
    const actual = df.select(
      col("data")
        .str
        .jsonPathMatch("$.os")
        .as("os")
    );
    expect(actual).toFrameEqual(expected);
    expect(seriesActual).toFrameEqual(expected);

  });
  test("lengths", () => {
    const df = pl.DataFrame({"os": ["kali", "debian", "ubuntu"]});
    const expected = pl.DataFrame({
      "lengths": [4, 6, 6]
    });
    const seriesActual = df.getColumn("os")
      .str
      .lengths()
      .rename("lengths")
      .toFrame();
    const actual = df.select(
      col("os")
        .str
        .lengths()
        .as("lengths")
    );
    expect(actual).toFrameEqual(expected);
    expect(seriesActual).toFrameEqual(expected);

  });
  test("replace", () => {
    const df = pl.DataFrame({
      "os": [
        "kali-linux",
        "debian-linux",
        "ubuntu-linux",
        "mac-sierra"
      ]
    });
    const expected = pl.DataFrame({
      "os": [
        "kali:linux",
        "debian:linux",
        "ubuntu:linux",
        "mac:sierra"
      ]
    });
    const seriesActual = df.getColumn("os")
      .str
      .replace("-", ":")
      .rename("os")
      .toFrame();
    const actual = df.select(
      col("os")
        .str
        .replace("-", ":")
        .as("os")
    );
    expect(actual).toFrameEqual(expected);
    expect(seriesActual).toFrameEqual(expected);
  });
  test("replaceAll", () => {
    const df = pl.DataFrame({
      "os": [
        "kali-linux-2021.3a",
        "debian-linux-stable",
        "ubuntu-linux-16.04",
        "mac-sierra-10.12.1"
      ]
    });
    const expected = pl.DataFrame({
      "os": [
        "kali:linux:2021.3a",
        "debian:linux:stable",
        "ubuntu:linux:16.04",
        "mac:sierra:10.12.1"
      ]
    });
    const seriesActual = df.getColumn("os")
      .str
      .replaceAll("-", ":")
      .rename("os")
      .toFrame();
    const actual = df.select(
      col("os")
        .str
        .replaceAll("-", ":")
        .as("os")
    );
    expect(actual).toFrameEqual(expected);
    expect(seriesActual).toFrameEqual(expected);
  });
  test("slice", () => {
    const df = pl.DataFrame({"os": ["linux-kali", "linux-debian", "windows-vista"]});
    const expected = pl.DataFrame({
      "first5": ["linux", "linux", "windo"]
    });
    const seriesActual = df.getColumn("os")
      .str
      .slice(0, 5)
      .rename("first5")
      .toFrame();
    const actual = df.select(
      col("os")
        .str
        .slice(0, 5)
        .as("first5")
    );
    expect(actual).toFrameEqual(expected);
    expect(seriesActual).toFrameEqual(expected);
  });
  test("strftime", () => {
    const df = pl.DataFrame({
      "timestamp": [
        "2020-01-01T01:22:00.002+00:00",
        "2020-02-01T01:02:01.030+00:00",
        "2021-11-01T01:02:20.001+00:00"
      ]
    });
    const expected = pl.DataFrame([
      pl.Series("datetime", [
        new Date(Date.parse("2020-01-01T01:22:00.002+00:00")),
        new Date(Date.parse("2020-02-01T01:02:01.030+00:00")),
        new Date(Date.parse("2021-11-01T01:02:20.001+00:00")),
      ]),
      pl.Series("date", [
        new Date(Date.parse("2020-01-01T01:22:00.002+00:00")),
        new Date(Date.parse("2020-02-01T01:02:01.030+00:00")),
        new Date(Date.parse("2021-11-01T01:02:20.001+00:00")),
      ], pl.Date)
    ]);

    const datetimeSeries = df.getColumn("timestamp")
      .str
      .strftime(pl.Datetime, "%FT%T%.3f%:z")
      .rename("datetime");
    const dateSeries = df.getColumn("timestamp")
      .str
      .strftime(pl.Date, "%FT%T%.3f%:z")
      .rename("date");

    const actualFromSeries = pl.DataFrame([datetimeSeries, dateSeries]);

    const actual = df.select(
      col("timestamp")
        .str
        .strftime(pl.Datetime, "%FT%T%.3f%:z")
        .as("datetime"),
      col("timestamp")
        .str
        .strftime(pl.Date, "%FT%T%.3f%:z")
        .as("date")
    );

    expect(actual).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(expected);
  });
  test("toLowercase", () => {
    const df = pl.DataFrame({
      "os": [
        "Kali-Linux",
        "Debian-Linux",
        "Ubuntu-Linux",
        "Mac-Sierra"
      ]
    });
    const expected = pl.DataFrame({
      "os": [
        "kali-linux",
        "debian-linux",
        "ubuntu-linux",
        "mac-sierra"
      ]
    });
    const seriesActual = df.getColumn("os")
      .str
      .toLowerCase()
      .rename("os")
      .toFrame();
    const actual = df.select(
      col("os")
        .str
        .toLowerCase()
        .as("os")
    );
    expect(actual).toFrameEqual(expected);
    expect(seriesActual).toFrameEqual(expected);
  });
  test("toUpperCase", () => {
    const df = pl.DataFrame({
      "os": [
        "Kali-Linux",
        "Debian-Linux",
        "Ubuntu-Linux",
        "Mac-Sierra"
      ]
    });
    const expected = pl.DataFrame({
      "os": [
        "KALI-LINUX",
        "DEBIAN-LINUX",
        "UBUNTU-LINUX",
        "MAC-SIERRA"
      ]
    });
    const seriesActual = df.getColumn("os")
      .str
      .toUpperCase()
      .rename("os")
      .toFrame();
    const actual = df.select(
      col("os")
        .str
        .toUpperCase()
        .as("os")
    );
    expect(actual).toFrameEqual(expected);
    expect(seriesActual).toFrameEqual(expected);
  });

});
describe("expr.lst", () => {
  test("get", () => {
    const df = pl.DataFrame({"a": [[1, 10, 11], [2, 10, 12], [1]]});
    const expected = pl.DataFrame({"get": [11, 12, null]});
    const actual = df.select(
      col("a")
        .lst
        .get(2)
        .as("get")
    );
    const actualFromSeries = df.getColumn("a")
      .lst
      .get(2)
      .rename("get")
      .toFrame();

    expect(actual).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(expected);
  });
  test("first", () => {
    const df = pl.DataFrame({"a": [[1, 10], [2, 10]]});
    const expected = pl.DataFrame({"first": [1, 2]});
    const actual = df.select(
      col("a")
        .lst
        .first()
        .as("first")
    );
    const actualFromSeries = df.getColumn("a")
      .lst
      .first()
      .rename("first")
      .toFrame();

    expect(actual).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(expected);
  });
  test("last", () => {
    const df = pl.DataFrame({"a": [[1, 10], [2, 12]]});
    const expected = pl.DataFrame({"last": [10, 12]});
    const actual = df.select(
      col("a")
        .lst
        .last()
        .as("last")
    );
    const actualFromSeries = df.getColumn("a")
      .lst
      .last()
      .rename("last")
      .toFrame();

    expect(actual).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(expected);
  });
  test("lengths", () => {
    const df = pl.DataFrame({"a": [[1], [2, 12], []]});
    const expected = pl.DataFrame({"lengths": [1, 2, 0]});
    const actual = df.select(
      col("a")
        .lst
        .lengths()
        .as("lengths")
    );
    const actualFromSeries = df.getColumn("a")
      .lst
      .lengths()
      .rename("lengths")
      .toFrame();

    expect(actual).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(expected);
  });
  test("max", () => {
    const df = pl.DataFrame({"a": [[1, -2], [2, 12, 1], [0, 1, 1, 5, 1]]});
    const expected = pl.DataFrame({"max": [1, 12, 5]});
    const actual = df.select(
      col("a")
        .lst
        .max()
        .as("max")
    );
    const actualFromSeries = df.getColumn("a")
      .lst
      .max()
      .rename("max")
      .toFrame();

    expect(actual).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(expected);
  });
  test("mean", () => {
    const df = pl.DataFrame({"a": [[1, -1], [2, 2, 8], [1, 1, 5, 1]]});
    const expected = pl.DataFrame({"mean": [0, 4, 2]});
    const actual = df.select(
      col("a")
        .lst
        .mean()
        .as("mean")
    );
    const actualFromSeries = df.getColumn("a")
      .lst
      .mean()
      .rename("mean")
      .toFrame();

    expect(actual).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(expected);
  });
  test("min", () => {
    const df = pl.DataFrame({"a": [[1, -2], [2, 12, 1], [0, 1, 1, 5, 1]]});
    const expected = pl.DataFrame({"min": [-2, 1, 0]});
    const actual = df.select(
      col("a")
        .lst
        .min()
        .as("min")
    );
    const actualFromSeries = df.getColumn("a")
      .lst
      .min()
      .rename("min")
      .toFrame();

    expect(actual).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(expected);
  });
  test("reverse", () => {
    const df = pl.DataFrame({"a": [[1, 2], [2, 0, 1], [0, 1, 1, 5, 1]]});
    const expected = pl.DataFrame({"reverse": [[2, 1], [1, 0, 2], [1, 5, 1, 1, 0]]});
    const actual = df.select(
      col("a")
        .lst
        .reverse()
        .as("reverse")
    );
    const actualFromSeries = df.getColumn("a")
      .lst
      .reverse()
      .rename("reverse")
      .toFrame();

    expect(actual).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(expected);
  });
  test("sort", () => {
    const df = pl.DataFrame({
      "a": [
        [1, 2, 1],
        [2, 0, 1],
        [0, 1, 1, 5, 1]
      ]
    });
    const expected = pl.DataFrame({
      "sort": [
        [1, 1, 2],
        [0, 1, 2],
        [0, 1, 1, 1, 5]
      ],
      "sort:reverse": [
        [2, 1, 1],
        [2, 1, 0],
        [5, 1, 1, 1, 0]
      ]
    });
    const actual = df.select(
      col("a")
        .lst
        .sort()
        .as("sort"),
      col("a")
        .lst
        .sort({reverse:true})
        .as("sort:reverse"),
    );

    const sortSeries = df.getColumn("a")
      .lst
      .sort()
      .rename("sort");

    const sortReverseSeries = df.getColumn("a")
      .lst
      .sort(true)
      .rename("sort:reverse");

    const actualFromSeries = pl.DataFrame([sortSeries, sortReverseSeries]);

    expect(actual).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(actual);
  });
  test("sum", () => {
    const df = pl.DataFrame({"a": [[1, 2], [2, 0, 1], [0, 1, 1, 5, 1]]});
    const expected = pl.DataFrame({"sum": [3, 3, 8]});
    const actual = df.select(
      col("a")
        .lst
        .sum()
        .as("sum")
    );
    const actualFromSeries = df.getColumn("a")
      .lst
      .sum()
      .rename("sum")
      .toFrame();

    expect(actual).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(actual);
  });
  test("unique", () => {
    const df = pl.DataFrame({"a": [[1, 2, 1], [2, 0, 1], [5, 5, 5, 5]]});
    const expected = pl.DataFrame({"unique": [[1, 2], [0, 1, 2], [5]]});
    const actual = df.select(
      col("a")
        .lst
        .unique()
        .lst
        .sort()
        .as("unique")
    );
    const actualFromSeries = df.getColumn("a")
      .lst
      .unique()
      .lst
      .sort()
      .rename("unique")
      .toFrame();

    expect(actual).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(actual);
  });
});
describe("expr.dt", () => {
  test("day", () => {
    const dt = new Date(Date.parse("08 Jan 84 01:10:30 UTC"));
    const df = pl.DataFrame([
      pl.Series("date_col", [dt], pl.Datetime)
    ]);
    const expected = pl.DataFrame({
      millisecond: [0],
      second: [30],
      minute: [10],
      hour: [1],
      day: [8],
      ordinalDay: [8],
      weekday: [6],
      week: [1],
      month: [1],
      year: [1984]
    });
    const dtCol = col("date_col").date;
    const dtSeries = df.getColumn("date_col").date;
    const actual = df.select(
      dtCol.nanosecond().as("millisecond"),
      dtCol.second().as("second"),
      dtCol.minute().as("minute"),
      dtCol.hour().as("hour"),
      dtCol.day().as("day"),
      dtCol.ordinalDay().as("ordinalDay"),
      dtCol.weekday().as("weekday"),
      dtCol.week().as("week"),
      dtCol.month().as("month"),
      dtCol.year().as("year")
    );

    const actualFromSeries = pl.DataFrame([
      dtSeries.nanosecond().rename("millisecond"),
      dtSeries.second().rename("second"),
      dtSeries.minute().rename("minute"),
      dtSeries.hour().rename("hour"),
      dtSeries.day().rename("day"),
      dtSeries.ordinalDay().rename("ordinalDay"),
      dtSeries.weekday().rename("weekday"),
      dtSeries.week().rename("week"),
      dtSeries.month().rename("month"),
      dtSeries.year().rename("year")
    ]);
    expect(actual).toFrameEqual(expected);
    expect(actualFromSeries).toFrameEqual(expected);
  });
});
