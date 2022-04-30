import * as pl from './pkg/index.js'
await pl.default()
await pl.init_hooks()
await pl.initThreadPool(navigator.hardwareConcurrency);
const res = await fetch("https://raw.githubusercontent.com/universalmind303/js-polars/js-polars-try-again/examples/1k.json")
const b = await res.arrayBuffer()
let df = pl.DataFrame.read_json(new Int8Array(b))
console.log(df.toRecords())
