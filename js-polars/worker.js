
import * as Comlink from 'comlink';
import * as pl from './pkg/polars.js'

self.onmessage = async event => {
  await pl.default()
  await pl.init_hooks()
  await pl.initThreadPool(2);
  console.log(pl)
  self.console.log("from worker")
  const randomArr = () => Array.from({length: 100_000}, () => Math.round((Math.random() * 100) % 20));

  const s1 = pl.Series.new_f64("a", randomArr())
  const s2 = pl.Series.new_f64("b", randomArr())
  let df = pl.DataFrame.read_columns([s1, s2][Symbol.iterator]())
  const s3 = pl.Series.new_f64("c", randomArr())

  df = df.add(s3)
  console.log(df.as_str())
};


