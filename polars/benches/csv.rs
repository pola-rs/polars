use criterion::{criterion_group, criterion_main, Criterion};
use polars::prelude::*;
use std::fs::File;

fn prepare_reader() -> Result<CsvReader<'static, File>> {
    let path =
        std::env::var("CSV_SRC").expect("env var CSV_SRC pointing to the csv_file is not set");

    Ok(CsvReader::from_path(&path)?.with_stop_after_n_rows(Some(10000)))
}

fn csv_parsing_benchmark(c: &mut Criterion) {
    c.bench_function("parse csv", |b| {
        b.iter(|| {
            let reader = prepare_reader().expect("file does not exist?");
            reader.finish().unwrap();
        })
    });
}

criterion_group!(benches, csv_parsing_benchmark);
criterion_main!(benches);
