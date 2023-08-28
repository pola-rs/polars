const pl = require("nodejs-polars");

// --8<-- [start:dataframe]
let df = pl.DataFrame({
  integer: [1, 2, 3],
  date: [
    new Date(2022, 1, 1, 0, 0),
    new Date(2022, 1, 2, 0, 0),
    new Date(2022, 1, 3, 0, 0),
  ],
  float: [4.0, 5.0, 6.0],
});
console.log(df);
// --8<-- [end:dataframe]

// --8<-- [start:csv]
df.writeCSV("output.csv");
var df_csv = pl.readCSV("output.csv");
console.log(df_csv);
// --8<-- [end:csv]

// --8<-- [start:csv2]
var df_csv = pl.readCSV("output.csv", { parseDates: true });
console.log(df_csv);
// --8<-- [end:csv2]

// --8<-- [start:json]
df.writeJSON("output.json", { format: "json" });
let df_json = pl.readJSON("output.json");
console.log(df_json);
// --8<-- [end:json]

// --8<-- [start:parquet]
df.writeParquet("output.parquet");
let df_parquet = pl.readParquet("output.parquet");
console.log(df_parquet);
// --8<-- [end:parquet]
