const pln = require("./index")
const pl = require('../nodejs-polars/bin/index')
const {Readable} = require('stream');
const util = require('util');

const p = "/home/cgrinstead/Development/pl_import/data/txns/100.csv"
// const buf = fs.readFileSync(p);
// console.log(pl.scanCsv(p).collect())
const fs = require('fs')
console.time("new")
const df = pln.scanCsv(p, {
  inferSchemaLength: 100,
  cache: true,
  hasHeader: true,
  ignoreErrors: true,
  skipRows: 0,
  sep: ',',
  rechunk: false,
  encoding: 'utf8',
  lowMemory: false,
  parseDates: false,
  skipRowsAfterHeader: 0
}).collectSync().rechunk()
console.log(df.estimatedSize())
const b = Buffer.alloc(df.estimatedSize())
df.writeCsv((err, values) => values, {})
df.write(b, {})
console.log(b)
// df.writeCsvCallback((err, values) => {
//   console.log("value=", values.toString())
// }, {})
// console.log(df.toObjects())
// const rows = util.promisify(df.toRowsCb)
// const bindRows = rows.bind(df)
// bindRows().then(data => console.log({data})).then(next => console.log({next}))