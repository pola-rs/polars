import pl from "./bin/index";
import fs from "fs";
console.time("csv");
let df = pl.readCSV("/home/cgrinstead/Development/pl_import/data/txns/10k.csv", {inferSchemaLength: 100, rechunk: true, batchSize: 1000});
console.timeEnd("csv");
console.log(df.nChunks());
console.time("ndjson");
df = pl.readJSON("/home/cgrinstead/Development/pl_import/data/txns/10k.json", {inferSchemaLength: 100, batchSize: 1000});
console.timeEnd("ndjson");
console.log(df.nChunks());

df = df.withColumn(pl.col("nonce").cast(pl.Float64))
  .withColumn(pl.col("block_number").cast(pl.Float64))
  .withColumn(pl.col("transaction_index").cast(pl.Float64))
  .withColumn(pl.col("value").cast(pl.Float64))
  .withColumn(pl.col("gas").cast(pl.Float64))
  .withColumn(pl.col("gas_price").cast(pl.Float64))
  .withColumn(pl.col("block_timestamp").cast(pl.Float64))
  .withColumn(pl.col("transaction_type").cast(pl.Float64));

const rows = df.toObject({orient: "row"});
const rowsValues = df.rows();


console.time("rows");
df = (pl.DataFrame as any).fromRowArrays(rows, 10);
console.timeEnd("rows");
console.log(df.nChunks());

console.time("rows_values");
df = pl.DataFrame(rowsValues, {orient: "row"});
console.timeEnd("rows_values");
console.log(df.nChunks());
