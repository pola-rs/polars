async function init() {
  const worker = new Worker("worker.js", {type: 'module'});
}
init();

// import('./dist/main.mjs').then(async (pl) => {
//   await pl.default()
//   await pl.init_hooks()
//   const res = await fetch("https://raw.githubusercontent.com/universalmind303/js-polars/js-polars-try-again/examples/1k.json")
//   const b = await res.arrayBuffer()
//   let df = pl.DataFrame.read_json(new Int8Array(b)).select(["hash", "from_address"]).unique(true, ['from_address'], 'first')
//   console.log(df.toRecords())
// })