use polars::io::mmap::MmapBytesReader;
use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let file = std::fs::File::open("/home/ritchie46/Downloads/tpch/tables_scale_100/lineitem.tbl")
        .unwrap();
    let file = Box::new(file) as Box<dyn MmapBytesReader>;
    let _df = CsvReader::new(file)
        .with_separator(b'|')
        .has_header(false)
        .with_chunk_size(10)
        .batched_mmap(None)
        .unwrap();

    // write_other_formats(&mut df)?;
    Ok(())
}

fn _write_other_formats(df: &mut DataFrame) -> PolarsResult<()> {
    let parquet_out = "../datasets/foods1.parquet";
    if std::fs::metadata(parquet_out).is_err() {
        let f = std::fs::File::create(parquet_out).unwrap();
        ParquetWriter::new(f).with_statistics(true).finish(df)?;
    }
    let ipc_out = "../datasets/foods1.ipc";
    if std::fs::metadata(ipc_out).is_err() {
        let f = std::fs::File::create(ipc_out).unwrap();
        IpcWriter::new(f).finish(df)?
    }
    Ok(())
}
