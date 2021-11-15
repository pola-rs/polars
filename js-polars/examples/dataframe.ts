import pl from '../polars'

const df = pl.Dataframe({
  num_col: [1,2],
  str_col: ["foo","bar"],
  date_col: [new Date(Date.now()), new Date(Date.now())],
  bool_col: [true, null],

})

console.log({
  shape: df.shape(),
  width: df.width(),
  height: df.height(),
  isEmpty: df.isEmpty(),
  df
})

