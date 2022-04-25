import * as pl from "./lib/polars.js";

await pl.default()
const col1 = pl.Series.new_f64("col_1", [1, 2, 3]);
const col2 = pl.Series.new_f64("col_2", [1, 2, 3]);
console.log(col1.sum())

document.getElementById('col1').innerHTML = col1.sum()
