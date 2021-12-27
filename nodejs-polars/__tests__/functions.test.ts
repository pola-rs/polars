import pl from "@polars";

describe("concat", () => {
  it("can concat multiple dataframes vertically", () => {
    const df1 = pl.DataFrame({
      "a": [1, 2, 3],
      "b": ["a", "b", "c"]
    });
    const df2 = pl.DataFrame({
      "a": [4, 5, 6],
      "b": ["d", "e", "f"]
    });
    const actual = pl.concat([df1, df2]);
    const expected = pl.DataFrame({
      "a": [1, 2, 3, 4, 5, 6],
      "b": ["a", "b", "c", "d", "e", "f"]
    });
    expect(actual).toFrameEqual(expected);
  });

  it("can concat multiple series vertically", () => {
    const s1 = pl.Series("a", [1, 2, 3]);
    const s2 = pl.Series("a", [4, 5, 6]);
    const actual = pl.concat([s1, s2]);
    const expected = pl.Series("a", [1, 2, 3, 4, 5, 6]);

    expect(actual).toSeriesEqual(expected);
  });
  it("cant concat empty list", () => {
    const fn = () => pl.concat([]);
    expect(fn).toThrowError();
  });

  it("can only concat series and df", () => {
    const fn = () => pl.concat([[1] as any, [2] as any]);
    expect(fn).toThrowError();
  });
  test("horizontal concat", () => {
    const a = pl.DataFrame({"a": ["a", "b"], "b": [1, 2]});
    const b = pl.DataFrame({"c": [5, 7, 8, 9], "d": [1, 2, 1, 2], "e": [1, 2, 1, 2]});
    const actual = pl.concat([a, b], {how:"horizontal"});
    const expected = pl.DataFrame(
      {
        "a": ["a", "b", null, null],
        "b": [1, 2, null, null],
        "c": [5, 7, 8, 9],
        "d": [1, 2, 1, 2],
        "e": [1, 2, 1, 2],
      }
    );
    expect(actual).toFrameEqual(expected);
  });
});
describe("repeat", () => {
  it("repeats a value n number of times into a series", () => {
    const value = "foobar";
    const actual = pl.repeat(value, 4, "foo");
    const expected = pl.Series("foo", Array.from({length: 4}, () => value));

    expect(actual).toSeriesEqual(expected);
  });
});
