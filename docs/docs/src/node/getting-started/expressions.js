const pl = require("nodejs-polars");

df = pl.DataFrame({
  a: [...Array(8).keys()],
  b: Array.from({ length: 8 }, () => Math.random()),
  c: [
    new Date(2022, 12, 1, 0, 0),
    new Date(2022, 12, 2, 0, 0),
    new Date(2022, 12, 3, 0, 0),
    new Date(2022, 12, 4, 0, 0),
    new Date(2022, 12, 5, 0, 0),
    new Date(2022, 12, 6, 0, 0),
    new Date(2022, 12, 7, 0, 0),
    new Date(2022, 12, 8, 0, 0),
  ],
  d: [1, 2.0, null, null, 0, -5, -42, null],
});

// --8<-- [start:select]
df.select(pl.col("*"));
// --8<-- [end:select]

// --8<-- [start:select2]
df.select(pl.col(["a", "b"]));
// --8<-- [end:select2]

// --8<-- [start:select3]
df.select([pl.col("a"), pl.col("b")]).limit(3);
// --8<-- [end:select3]

// --8<-- [start:exclude]
df.select([pl.exclude("a")]);
// --8<-- [end:exclude]

// --8<-- [start:filter]
df.filter(pl.col("c").gt(new Date(2022, 12, 2)).lt(new Date(2022, 12, 8)));
// --8<-- [end:filter]

// --8<-- [start:filter2]
df.filter(pl.col("a").ltEq(3).and(pl.col("d").isNotNull()));
// --8<-- [end:filter2]

// --8<-- [start:with_columns]
df.withColumns([
  pl.col("b").sum().alias("e"),
  pl.col("b").plus(42).alias("b+42"),
]);
// --8<-- [end:with_columns]

// --8<-- [start:dataframe2]
df2 = pl.DataFrame({
  x: [...Array(8).keys()],
  y: ["A", "A", "A", "B", "B", "C", "X", "X"],
});
// --8<-- [end:dataframe2]

// --8<-- [start:groupby]
df2.groupBy("y").count();
console.log(df2);
// --8<-- [end:groupby]

// --8<-- [start:groupby2]
df2
  .groupBy("y")
  .agg(pl.col("*").count().alias("count"), pl.col("*").sum().alias("sum"));
// --8<-- [end:groupby2]

// --8<-- [start:combine]
df_x = df
  .withColumns(pl.col("a").mul(pl.col("b")).alias("a * b"))
  .select([pl.all().exclude(["c", "d"])]);

console.log(df_x);
// --8<-- [end:combine]

// --8<-- [start:combine2]
df_y = df
  .withColumns([pl.col("a").mul(pl.col("b")).alias("a * b")])
  .select([pl.all().exclude("d")]);
console.log(df_y);
// --8<-- [end:combine2]
