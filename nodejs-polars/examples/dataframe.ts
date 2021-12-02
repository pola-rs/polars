/* eslint-disable no-undef */
import pl from "../polars";

const jsonpath = "/home/cgrinstead/Development/git/covalent/blocks.json";
const csvpath = "/home/cgrinstead/Development/git/covalent/blocks.csv";
const ldf = pl.readCSV(csvpath)
  .lazy()
  .select(
    pl.col("hash").alias("num_hashes")
      .nUnique(),
    pl.col("number").alias("max_block_height")
      .max(),
    pl.col("gas_used").alias("gas"),
    pl.col("transaction_count").mean(),
  )
  .collectSync();