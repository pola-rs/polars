// --8<-- [start:setup]
const pl = require("nodejs-polars");
const Chance = require("chance");

const chance = new Chance(42);
// --8<-- [end:setup]

// --8<-- [start:dataframe]

const arr = Array.from({ length: 5 }).map((_) =>
  chance.floating({ min: 0, max: 1 }),
);

let df = pl.DataFrame({
  nrs: [1, 2, 3, null, 5],
  names: ["foo", "ham", "spam", "egg", null],
  random: arr,
  groups: ["A", "A", "B", "C", "B"],
});
console.log(df);
// --8<-- [end:dataframe]

// --8<-- [start:select]
let out = df.select(
  pl.col("nrs").sum(),
  pl.col("names").sort(),
  pl.col("names").first().alias("first name"),
  pl.mean("nrs").multiplyBy(10).alias("10xnrs"),
);
console.log(out);
// --8<-- [end:select]

// --8<-- [start:filter]
out = df.filter(pl.col("nrs").gt(2));
console.log(out);
// --8<-- [end:filter]

// --8<-- [start:with_columns]

df = df.withColumns(
  pl.col("nrs").sum().alias("nrs_sum"),
  pl.col("random").count().alias("count"),
);

console.log(df);
// --8<-- [end:with_columns]

// --8<-- [start:groupby]
out = df.groupBy("groups").agg(
  pl
    .col("nrs")
    .sum(), // sum nrs by groups
  pl
    .col("random")
    .count()
    .alias("count"), // count group members
  // sum random where name != null
  pl
    .col("random")
    .filter(pl.col("names").isNotNull())
    .sum()
    .suffix("_sum"),
  pl.col("names").reverse().alias("reversed names"),
);
console.log(out);
// --8<-- [end:groupby]
