#[macro_use]
extern crate criterion;
use criterion::Criterion;

use polars::prelude::*;

fn bench_sort(s: &Int32Chunked) {
    criterion::black_box(s.sort(false));
}

fn add_benchmark(c: &mut Criterion) {
    (10..=20).step_by(2).for_each(|log2_size| {
        let size = 2usize.pow(log2_size);

        let ca = Int32Chunked::init_rand(size, 0.0, 10);

        c.bench_function(&format!("sort 2^{} i32", log2_size), |b| {
            b.iter(|| bench_sort(&ca))
        });

        let ca = Int32Chunked::init_rand(size, 0.1, 10);
        c.bench_function(&format!("sort null 2^{} i32", log2_size), |b| {
            b.iter(|| bench_sort(&ca))
        });
    });
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
