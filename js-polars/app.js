const polars_internal = require("./bin/libpolars.node");

const Dataframe = (df) => {
  const head = (length = 5) => Dataframe.from(polars_internal.head({_df: df, length}));
  const show = (length = 5) => Dataframe.from(polars_internal.show({_df: df, length}));

  return {
    head,
    show
  }
}

Dataframe.from = (df) => Dataframe(df)
const read_csv = (path) => Dataframe.from(polars_internal.read_csv({path}))

let path = "../examples/aggregate_multiple_files_in_chunks/datasets/foods5.csv"
let df = read_csv(path)
df = df.head(10)
df.show()
df = df.head(1)
df.show()



