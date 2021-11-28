import pl from "@polars";

describe('groupby', () => {
  it('can perform a basic groupby', () => {
    const  df = pl.DataFrame({
      "a": ["a", "b", "a", "b", "b", "c"],
      "b": [1, 1, 1, 1, 5, 6],
      "c": [6, 5, 4, 3, 2, 1],
    });

  });
});