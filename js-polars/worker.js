import * as pl from './pkg/index.js'
const filepath = "http://localhost:8000/examples/reddit_1m.csv"

await pl.default()
await pl.init_hooks()
await pl.initThreadPool(navigator.hardwareConcurrency);

const res = await fetch(filepath)
const b = await res.arrayBuffer()
console.time("readCsv")
const infer_schema_length = 10;
const chunk_size = 40000
const has_header = true
const ignore_errors = true;
const n_rows = undefined;
const skip_rows = 0;
const rechunk = false;
const encoding = 'utf8';
const n_threads = 4;
const low_memory = false;
const parse_dates = true;
const skip_rows_after_header = 0
const df = pl.read_csv(
  new Int8Array(b),
  infer_schema_length,
  chunk_size,
  has_header,
  ignore_errors,
  n_rows,
  skip_rows,
  rechunk,
  encoding,
  n_threads,
  low_memory,
  parse_dates,
  skip_rows_after_header,
).rechunk().head(100)
console.timeEnd("readCsv")

console.time('toRecords')
df.toRecords()
console.timeEnd('toRecords')

self.postMessage(df.head().toObject())