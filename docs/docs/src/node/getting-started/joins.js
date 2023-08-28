const pl = require("nodejs-polars");

// --8<-- [start:join]
df = pl.DataFrame({
  a: [...Array(8).keys()],
  b: Array.from({ length: 8 }, () => Math.random()),
  d: [1, 2.0, null, null, 0, -5, -42, null],
});

df2 = pl.DataFrame({
  x: [...Array(8).keys()],
  y: ["A", "A", "A", "B", "B", "C", "X", "X"],
});
joined = df.join(df2, { leftOn: "a", rightOn: "x" });
console.log(joined);
// --8<-- [end:join]

// --8<-- [start:hstack]
stacked = df.hstack(df2);
console.log(stacked);
// --8<-- [end:hstack]
