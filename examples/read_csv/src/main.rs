use polars::prelude::*;

fn main() -> Result<()> {
    let mut df = LazyCsvReader::new("../datasets/foods1.csv")
        .finish()?
        .select([
            // select all columns
            all(),
            // and do some aggregations
            cols(["fats_g", "sugars_g"]).sum().suffix("_summed"),
        ])
        .collect()?;

    dbg!(&df);

    write_other_formats(&mut df)?;
    Ok(())
}

fn write_other_formats(df: &mut DataFrame) -> Result<()> {
    let parquet_out = "../datasets/foods1.parquet";
    if std::fs::metadata(&parquet_out).is_err() {
        let f = std::fs::File::create(&parquet_out).unwrap();
        ParquetWriter::new(f).with_statistics(true).finish(df)?;
    }
    let ipc_out = "../datasets/foods1.ipc";
    if std::fs::metadata(&ipc_out).is_err() {
        let f = std::fs::File::create(&ipc_out).unwrap();
        IpcWriter::new(f).finish(df)?
    }
    Ok(())
}
