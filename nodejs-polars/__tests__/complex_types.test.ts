import pl from "@polars";


describe("complex types", () => {

  test.skip("nested arrays round trip", () => {
    const arr = [[["foo"]], [], null];
    const s = pl.Series("", arr);
    const actual = s.toArray();
    expect(actual).toEqual(arr);
  });
  test.skip("struct arrays round trip", () => {
    const arr = [{foo: "a", bar: 1}, null, null];
    const s = pl.Series("", arr);
    const actual = s.toArray();
    expect(actual).toEqual(arr);
  });

});
