import pl from '../polars'

const df = pl.Dataframe([
  {
    str_col: "apples",
    num_col: 123,
    bool_col: false,
    date_col: new Date(Date.now()),
    nullable_col: null,
    arr_col: ["foo", "bar", "baz"],
    buff_type: Buffer.from([102, 111, 111] as any, "utf-8")
  },
  {
    str_col: "oranges",
    num_col: 123,
    bool_col: null,
    date_col: new Date(Date.now()),
    nullable_col: "Hello",
    arr_col: ["foo", "bar", "baz"],
    buff_type: Buffer.from("foo-2", "utf-8")
  },
  {
    str_col: "pineapples",
    num_col: 123.1123,
    bool_col: true,
    date_col: new Date(Date.now()),
    nullable_col: undefined,
    arr_col: ["foo", "bar", "baz"],
    buff_type: Buffer.from("foo-3", "utf-8")
  },
])


console.log({
  shape: df.shape(),
  width: df.width(),
  height: df.height(),
  isEmpty: df.isEmpty(),
  df
})