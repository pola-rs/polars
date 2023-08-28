const pl = require("nodejs-polars");

// --8<-- [start:eager]

df = pl.readCSV("docs/src/data/iris.csv");
df_small = df.filter(pl.col("sepal_length").gt(5));
df_agg = df_small.groupBy("species").agg(pl.col("sepal_width").mean());
console.log(df_agg);
// --8<-- [end:eager]

// --8<-- [start:lazy]
q = pl
  .scanCSV("docs/src/data/iris.csv")
  .filter(pl.col("sepal_length").gt(5))
  .groupBy("species")
  .agg(pl.col("sepal_width").mean());

df = q.collect();
// --8<-- [end:lazy]
