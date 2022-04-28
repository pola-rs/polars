import pl from "@polars";
describe("json", () => {
  test("lazyframe", () => {
    const df = pl.DataFrame({
      foo: [1, 2, 3]
    })
      .lazy()
      .filter(pl.col("foo").greaterThan(1));

    const s = JSON.stringify(df);
    console.log(s);

  });
});
