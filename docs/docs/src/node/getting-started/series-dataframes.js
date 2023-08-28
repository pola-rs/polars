// --8<-- [start:series]
const pl = require("nodejs-polars");

var s = pl.Series("a", [1, 2, 3, 4, 5]);
console.log(s);
// --8<-- [end:series]

// --8<-- [start:minmax]
var s = pl.Series("a", [1, 2, 3, 4, 5]);
console.log(s.min());
console.log(s.max());
// --8<-- [end:minmax]

// --8<-- [start:string]
var s = pl.Series("a", ["polar", "bear", "arctic", "polar fox", "polar bear"]);
var s2 = s.str.replace("polar", "pola");
console.log(s2);
// --8<-- [end:string]

// --8<-- [start:dt]
var s = pl.Series("a", [
  new Date(2001, 1, 1),
  new Date(2001, 1, 3),
  new Date(2001, 1, 5),
  new Date(2001, 1, 7),
  new Date(2001, 1, 9),
]);
var s2 = s.date.day();
console.log(s2);
// --8<-- [end:dt]

// --8<-- [start:dataframe]
let df = pl.DataFrame({
  integer: [1, 2, 3, 4, 5],
  date: [
    new Date(2022, 1, 1, 0, 0),
    new Date(2022, 1, 2, 0, 0),
    new Date(2022, 1, 3, 0, 0),
    new Date(2022, 1, 4, 0, 0),
    new Date(2022, 1, 5, 0, 0),
  ],
  float: [4.0, 5.0, 6.0, 7.0, 8.0],
});
console.log(df);
// --8<-- [end:dataframe]

// --8<-- [start:head]
console.log(df.head(3));
// --8<-- [end:head]

// --8<-- [start:tail]
console.log(df.tail(3));
// --8<-- [end:tail]

// --8<-- [start:sample]
console.log(df.sample(2));
// --8<-- [end:sample]

// --8<-- [start:describe]
console.log(df.describe());
// --8<-- [end:describe]
