import * as pl from './pkg/polars.js';

self.onmessage = async event => {
  await pl.default()
  console.log("worker.js")
  console.log(pl)
  await pl.init_hooks()
  await pl.initThreadPool(navigator.hardwareConcurrency);
  const col1 = pl.Series.new_f64("col_1", [1, 2, 3]);
  console.log(col1.toString())
  const col2 = pl.Series.new_f64("col_2", [1, 2, 3]);
  const df = pl.DataFrame.read_columns([col1, col2][Symbol.iterator]());
  // console.log(df.get_columns())
  console.log(df.unique(true, ['col_1'], 'first'))
};
