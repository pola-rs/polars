use std::fs::File;

use criterion::{criterion_group, criterion_main, Criterion};
use polars::prelude::*;

fn csv_parsing_benchmark(c: &mut Criterion) {
    c.bench_function("stream_csv", |b| {
        b.iter(|| {
            LazyFrame::scan_parquet(
                "/home/ritchie46/Downloads/csv-benchmark/yellow_tripdata_2010-01.parquet",
                Default::default(),
            )
            .unwrap()
            .groupby([col("rate_code")])
            .agg([
                col("rate_code").sum(),
                col("rate_code").first().alias("first"),
            ])
            .with_streaming(true)
            .collect()
        })
    });
}

criterion_group!(benches, csv_parsing_benchmark);
criterion_main!(benches);
