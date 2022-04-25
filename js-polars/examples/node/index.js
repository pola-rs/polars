import * as pl from "../../pkg/node/polars.js";

const col1 = pl.Series.new_f64("col_1", [1, 2, 3]);
const col2 = pl.Series.new_f64("col_2", [1, 2, 3]);

const df = pl.DataFrame.read_columns([col1, col2][Symbol.iterator]());

df.columns = ["foo", "bar"];

console.log(df);
