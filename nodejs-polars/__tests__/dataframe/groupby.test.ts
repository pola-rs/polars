import pl from "@polars";

describe("groupby", () => {
  it("can perform a basic groupby", () => {
    const  df = pl.DataFrame({
      "a": ["a", "b", "a", "b", "b", "c"],
      "b": [1, 1, 1, 1, 5, 6],
      "c": [6, 5, 4, 3, 2, 1],
    });
    const actual = df
      .groupBy("a")
      .first()
      .sort("a");

    expect([...actual.getColumn("b_first")]).toStrictEqual([1,1,6]);
    expect([...actual.getColumn("c_first")]).toStrictEqual([6,5,1]);
  });
});