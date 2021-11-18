import pl from '../polars'

const df1 = pl.Dataframe({
  num_col: [1,2],
  str_col: ["foo","bar"],
  date_col: [new Date(Date.now()), new Date(Date.now())],
  bool_col: [true, null],
})
const df = pl.fromRows([
  {
    str_col: "apples",
    num_col: 123,
    bool_col: false,
    date_col: new Date(Date.now()),
    nullable_col: null,
    buff_type: Buffer.from([102, 111, 111] as any, "utf-8")
  },
  {
    str_col: "oranges",
    num_col: 123,
    bool_col: true,
    date_col: new Date(Date.now()),
    nullable_col: "Hello",
    buff_type: Buffer.from("foo-2", "utf-8")
  },
])
console.log({
  shape: df.shape(),
  width: df.width(),
  height: df.height(),
  isEmpty: df.isEmpty(),
  df
})

