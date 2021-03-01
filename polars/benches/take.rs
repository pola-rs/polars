use criterion::{criterion_group, criterion_main, Criterion};
use polars::prelude::*;
use polars_core::utils::split_ca;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn create_primitive_ca(size: u32, null_percentage: f32, n_chunks: usize) -> UInt32Chunked {
    let mut rng = StdRng::seed_from_u64(0);
    let ca: UInt32Chunked = (0..size)
        .map(|i| {
            if rng.gen::<f32>() < null_percentage {
                None
            } else {
                Some(i)
            }
        })
        .collect();

    match n_chunks {
        1 => ca,
        n => split_ca(&ca, n)
            .unwrap()
            .into_iter()
            .reduce(|mut acc, arr| {
                acc.append(&arr);
                acc
            })
            .unwrap(),
    }
}

fn create_random_idx(size: usize) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(0);
    (0..size).map(|_| rng.gen_range(0..size)).collect()
}

fn bench_take(ca: &UInt32Chunked, idx: &[usize]) {
    let f = || ca.take(idx.iter().copied().into());
    criterion::black_box(f());
}

fn add_benchmark(c: &mut Criterion) {
    let idx = create_random_idx(1024);
    let ca = create_primitive_ca(1024, 0.0, 1);
    c.bench_function("take primitive 1024 0% nulls array;", |b| {
        b.iter(|| bench_take(&ca, &idx))
    });
    let ca = create_primitive_ca(1024, 0.05, 1);
    c.bench_function("take primitive 1024 5% nulls array;", |b| {
        b.iter(|| bench_take(&ca, &idx))
    });
    let ca = create_primitive_ca(1024, 0.05, 3);
    c.bench_function("take primitive 1024 5% nulls array; 3 chunks", |b| {
        b.iter(|| bench_take(&ca, &idx))
    });

    let idx = create_random_idx(4096);
    let ca = create_primitive_ca(4096, 0.0, 1);
    c.bench_function("take primitive 4096 0% nulls array;", |b| {
        b.iter(|| bench_take(&ca, &idx))
    });
    let ca = create_primitive_ca(4096, 0.05, 1);
    c.bench_function("take primitive 4096 5% nulls array;", |b| {
        b.iter(|| bench_take(&ca, &idx))
    });
    let ca = create_primitive_ca(4096, 0.05, 3);
    c.bench_function("take primitive 4096 5% nulls array; 3 chunks", |b| {
        b.iter(|| bench_take(&ca, &idx))
    });
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
