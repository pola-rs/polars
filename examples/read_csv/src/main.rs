use polars::io::mmap::MmapBytesReader;
use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let file = std::fs::File::open("/home/ritchie46/Downloads/pdsh/tables_scale_100/lineitem.tbl")
        .unwrap();
    let file = Box::new(file) as Box<dyn MmapBytesReader>;
    let _df = CsvReadOptions::default()
        .map_parse_options(|x| x.with_separator(b'|'))
        .with_has_header(false)
        .with_chunk_size(10)
        .into_reader_with_file_handle(file);

    // write_other_formats(&mut df)?;
    Ok(())
}

fn _write_other_formats(df: &mut DataFrame) -> PolarsResult<()> {
    let parquet_out = "../datasets/foods1.parquet";
    if std::fs::metadata(parquet_out).is_err() {
        let f = std::fs::File::create(parquet_out).unwrap();
        ParquetWriter::new(f)
            .with_statistics(StatisticsOptions::full())
            .finish(df)?;
    }
    let ipc_out = "../datasets/foods1.ipc";
    if std::fs::metadata(ipc_out).is_err() {
        let f = std::fs::File::create(ipc_out).unwrap();
        IpcWriter::new(f).finish(df)?
    }
    Ok(())
}
