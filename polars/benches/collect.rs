use criterion::{criterion_group, criterion_main, Criterion};
use polars::prelude::*;

fn bench_collect(v: &[bool]) {
    let f = || v.iter().copied().collect::<ChunkedArray<_>>();
    criterion::black_box(f());
}

fn add_benchmark(c: &mut Criterion) {
    let v = vec![true; 1024];
    c.bench_function("collect bool 1024", |b| b.iter(|| bench_collect(&v)));
    let v = vec![true; 4096];
    c.bench_function("collect bool 4096", |b| b.iter(|| bench_collect(&v)));
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
