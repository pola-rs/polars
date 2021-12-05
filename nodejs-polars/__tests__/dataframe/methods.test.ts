import pl from "@polars";
import Chance from "chance";

describe("dataframe", () => {
  const chance = new Chance();
  const buildDF = () => pl.DataFrame({
    "num": [chance.natural(), chance.natural(), chance.natural()],
    "str": [chance.string(), chance.string(), chance.string()],
    "date": [chance.date(), chance.date(), chance.date()]
  });

  it("can clone", () => {
    const df = buildDF();
    const cloned = df.clone();
    expect(df.frameEqual(cloned)).toStrictEqual(true);
  });
  it.todo("can describe");

  it("can drop a column", () => {
    const df = buildDF();
    expect(df.drop("date").columns).toEqual(["num", "str"]);
  });

  it("can drop multiple columns", () => {
    const df = buildDF();
    expect(df.drop(["date", "num"]).columns).toEqual(["str"]);
  });
  it.todo("can drop duplicates");
  it.todo("can drop duplicates for only a subset of columns");

  it("can drop nulls", () => {
    const df = pl.DataFrame({
      "num": [chance.natural(), null, chance.natural()],
      "str": [chance.string(), chance.string(), chance.string()],
      "date": [chance.date(), chance.date(), chance.date()]
    }).dropNulls();
    expect(df.height).toStrictEqual(2);
  });

  it("can drop nulls for a given subset", () => {
    const df = pl.DataFrame({
      "num": [chance.natural(), null, chance.natural()],
      "str": [chance.string(), chance.string(), chance.string()],
      "date": [chance.date(), chance.date(), chance.date()],
      "num2": [chance.natural(), null, null],
    }).dropNulls("num");
    expect(df.height).toStrictEqual(2);
    const df2 = pl.DataFrame({
      "num": [chance.natural(), null, chance.natural()],
      "str": [chance.string(), chance.string(), chance.string()],
      "date": [chance.date(), chance.date(), chance.date()],
      "num2": [chance.natural(), null, null],
    }).dropNulls(["num", "str"]);
    expect(df2.height).toStrictEqual(2);
  });
  it.todo("can can explode");

  it.each`
  strategy      | expected
  ${"max"}      | ${5}
  ${"min"}      | ${1}
  ${"backward"} | ${1}
  ${"forward"}  | ${5}
  ${"max"}      | ${5}
  ${"zero"}     | ${0}
  ${"one"}      | ${1}
  `("can fill nulls with a given strategy", ({strategy, expected}) => {
    const df =buildDF().hstack([pl.Series("null_col", [5, null, 1])]);
    expect(df.fillNull(strategy).getColumn("null_col")[1]).toStrictEqual(expected);
  });
  it.todo("can filter");

  it("can findIdxByName", () => {
    const idx = buildDF().findIdxByName("date");
    expect(idx).toStrictEqual(2);
  });

  it.each`
  method        | expected
  ${"plus"}     | ${[5, 6, 8]}
  ${"minus"}    | ${[-1, -2, -4]}
  ${"times"}    | ${[4, 8, 16]}
  ${"divide"}    | ${[1, 0.5, 0.25]}
  `("can fold", ({method, expected}) => {
    const df = pl.DataFrame({
      "a": [2, 2, 2],
      "b": [2, 2, 2],
      "c": [1, 2, 4]
    });
    const s = df.fold((s1: any, s2: any) => s1[method](s2));
    expect([...s]).toEqual(expected);
  });

  it("can get a column", () => {
    expect(buildDF().getColumn("date").name).toStrictEqual("date");
  });

  it("can get multiple columns", () => {
    expect(
      buildDF()
        .getColumns()
        .map(s => s.name)
    )
      .toEqual(["num", "str", "date"]);

  });

  it("can get hash rows", () => {
    const series = buildDF().hashRows();
    expect(series.length).toStrictEqual(3);
    expect(series.dtype).toStrictEqual("UInt64");
  });

  it("can get the head", () => {
    const df = buildDF().head(1);
    expect(df.height).toStrictEqual(1);
  });

  it("can insert at an index", () => {
    const df = buildDF();
    const s = pl.Series("new_col", [1, 2, 3]);
    df.insertAtIdx(0, s);
    expect(df.findIdxByName("new_col")).toStrictEqual(0);
  });
  it.todo("can interpolate");

  it("can get tell you what rows are duplicated", () => {
    const series = pl.DataFrame({
      "a": [2, 2, 2, 2],
      "b": [2, 2, 2, 2],
      "c": [1, 2, 2, 4]
    }).isDuplicated();
    expect([...series]).toEqual([false, true, true, false]);
  });

  it("can get tell you if its empty", () => {
    const df = pl.DataFrame({});
    expect(df.isEmpty()).toStrictEqual(true);
  });

  it("can get tell you what rows are unique", () => {
    const series = pl.DataFrame({
      "a": [2, 2, 2, 2],
      "b": [2, 2, 2, 2],
      "c": [1, 2, 2, 4]
    }).isUnique();
    expect([...series]).toEqual([true, false, false, true]);
  });

  it("can perform inner join", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    const otherDF = pl.DataFrame({
      "apple": ["x", "y", "z"],
      "ham": ["a", "b", "d"]
    });
    const df3 = df.join(otherDF, {on: "ham"});
    expect(df3.shape).toStrictEqual({height: 2, width: 4});
  });

  it("can perform left join", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    const otherDF = pl.DataFrame({
      "apple": ["x", "y", "z"],
      "ham": ["a", "b", "d"]
    });
    const df3 = df.join(otherDF, {on: "ham", how: "left"});
    expect(df3.getColumn("apple")[2]).toStrictEqual(null);
  });

  it("can perform outer join", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    const otherDF = pl.DataFrame({
      "apple": ["x", "y", "z"],
      "ham": ["a", "b", "d"]
    });
    const df3 = df.join(otherDF, {on: "ham", how: "outer"});
    expect(df3.getColumn("apple")[3]).toStrictEqual(null);
    expect(df3.getColumn("foo")[2]).toStrictEqual(null);
    expect(df3.getColumn("bar")[2]).toStrictEqual(null);
  });

  it("can get the max of a dataframe", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).max();
    expect(df.rows()[0]).toEqual([3, 8, null]);
  });

  it("can get the min of a dataframe", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).min();
    expect(df.rows()[0]).toEqual([1, 6, null]);
  });
  it("can get the number of chunks", () => {});

  it("can get the null count", () => {
    const df = pl.DataFrame({
      "foo": [1, null, 3],
      "bar": [6, 7, null],
      "ham": ["a", "b", "c"]
    }).nullCount();
    expect(df.rows()[0]).toEqual([1, 1, 0]);
  });
  it.todo("can pipe");

  it("can get the quantile", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).quantile(0.5);
    expect(df.rows()[0]).toEqual([2, 7, null]);
  });

  it("can get number of chunks", () => {
    const chunks = pl.DataFrame({
      "foo": [1, null, 3],
      "bar": [6, 7, null],
      "ham": ["a", "b", "c"]
    }).nChunks();
    expect(chunks).toStrictEqual(1);
  });
});