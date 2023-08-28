const pl = require("nodejs-polars");

df = pl.DataFrame({
  foo: [1, 2, 3, null, 5],
  bar: ["foo", "ham", "spam", "egg", null],
});

// --8<-- [start:example1]
pl.col("foo").sort().head(2);
// --8<-- [end:example1]

// --8<-- [start:example2]
df.select(
  pl.col("foo").sort().head(2),
  pl.col("bar").filter(pl.col("foo").eq(1)).sum(),
);
// --8<-- [end:example2]
