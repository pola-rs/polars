
import pl from "@polars";

describe("struct", () => {
  test("series <--> array round trip", () => {
    const data = [
      {utf8: "a", f64: 1, },
      {utf8: "b", f64: 2, }
    ];
    const name = "struct";
    const s = pl.Series(name, data);
    expect(s.name).toEqual(name);
    expect(s.toArray()).toEqual(data);
  });
  test("pli.struct", () => {
    const expected = pl.DataFrame({
      foo: [1],
      bar: [2]
    }).toStruct("foo");
    const foo = pl.Series("foo", [1]);
    const bar = pl.Series("bar", [2]);
    const actual = pl.struct([foo, bar]).rename("foo");
    expect(actual).toSeriesEqual(expected);
  });
  test("pli.struct dataframe", () => {
    const df = pl.DataFrame({
      foo: [1],
      bar: [2]
    });
    const actual = df
      .select(pl.struct(pl.cols("foo", "bar")).alias("s"))
      .toSeries();
    expect(actual).toSeriesEqual(df.toStruct("s"));
  });
  test("struct toArray", () => {
    const actual = pl.DataFrame({
      foo: [1, 10, 100],
      bar: [2, null, 200]
    })
      .toStruct("foobar")
      .toArray();

    const expected = [
      {foo: 1, bar: 2},
      {foo: 10, bar: null},
      {foo: 100, bar: 200}
    ];
    expect(actual).toEqual(expected);
  });
});
