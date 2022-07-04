import pl from "@polars";
import path from "path";
describe("serde", () => {
  test.only("lazyframe:json", () => {
    const df = pl.scanCSV(path.resolve("../examples/datasets/foods1.csv"));
    const buf = df.serialize("json");
    console.log(buf.toString());
    const deserde = pl.LazyDataFrame.deserialize(buf, "json");
    console.log(deserde);
    // const expected = df.collectSync();
    const actual = deserde.collectSync();
    console.log(actual);
    // expect(actual).toFrameEqual(expected);
  });

  test.skip("lazyframe:bincode", () => {
    const df = pl.scanCSV("../examples/datasets/foods1.csv");
    const buf = df.serialize("bincode");
    const deserde = pl.LazyDataFrame.deserialize(buf, "bincode");
    const expected = df.collectSync();
    const actual = deserde.collectSync();
    expect(actual).toFrameEqual(expected);
  });
  test("expr:json", () => {
    const expr = pl.cols("foo", "bar").sortBy("other");

    const buf = expr.serialize("json");
    const actual = pl.Expr.deserialize(buf, "json");

    expect(actual.toString()).toEqual(expr.toString());
  });
  test("expr:bincode", () => {
    const expr = pl.cols("foo", "bar").sortBy("other");
    const buf = expr.serialize("bincode");
    const actual = pl.Expr.deserialize(buf, "bincode");

    expect(actual.toString()).toEqual(expr.toString());
  });
  test("dataframe:json", () => {
    const df = pl.DataFrame({
      foo: [1, 2],
      bar: [2, 3]
    });
    const buf = df.serialize("json");
    const expected = pl.DataFrame.deserialize(buf, "json");
    expect(df).toFrameEqual(expected);
  });
  test("dataframe:bincode", () => {
    const df = pl.DataFrame({
      foo: [1, 2],
      bar: [2, 3]
    });
    const buf = df.serialize("bincode");
    const expected = pl.DataFrame.deserialize(buf, "bincode");
    expect(df).toFrameEqual(expected);
  });

  test("dataframe:unsupported", () => {
    const df = pl.DataFrame({
      foo: [1, 2],
      bar: [2, 3]
    });
    const ser = () => df.serialize("yaml" as any);
    const buf = df.serialize("bincode");
    const de = () => pl.DataFrame.deserialize(buf, "yaml" as any);
    const mismatch = () => pl.DataFrame.deserialize(buf, "json");
    expect(ser).toThrow();
    expect(de).toThrow();
    expect(mismatch).toThrow();
  });
  test("series:json", () => {
    const s = pl.Series("foo", [1, 2, 3]);

    const buf = s.serialize("json");
    const expected = pl.Series.deserialize(buf, "json");
    expect(s).toSeriesEqual(expected);
  });
  test("series:bincode", () => {
    const s = pl.Series("foo", [1, 2, 3]);

    const buf = s.serialize("bincode");
    const expected = pl.Series.deserialize(buf, "bincode");
    expect(s).toSeriesEqual(expected);
  });

  test("series:unsupported", () => {
    const s = pl.Series("foo", [1, 2, 3]);
    const ser = () => s.serialize("yaml" as any);
    const buf = s.serialize("bincode");
    const de = () => pl.Series.deserialize(buf, "yaml" as any);
    const mismatch = () => pl.Series.deserialize(buf, "json");
    expect(ser).toThrow();
    expect(de).toThrow();
    expect(mismatch).toThrow();
  });
});
