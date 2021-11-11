import pl from './polars'

const path = "../examples/aggregate_multiple_files_in_chunks/datasets/foods5.csv"
let df = pl.readCsv(path)
let series = pl.Series({
  name: "foo", 
  values: ["bar", "baz", "barzz"]
})

console.log(series)
console.log({
  head: df.head(),
  shape: df.shape(),
  width: df.width(),
  height: df.height(),
  isEmpty: df.isEmpty(),
  df
})