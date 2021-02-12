use criterion::{criterion_group, criterion_main, Criterion};
use polars::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn bench_collect_bool(v: &[bool]) {
    let f = || v.iter().copied().collect::<ChunkedArray<_>>();
    criterion::black_box(f());
}

fn bench_collect_num(v: &[f32]) {
    let f = || v.iter().copied().collect::<NoNull<Float32Chunked>>();
    criterion::black_box(f());
}

fn bench_collect_optional_num(v: &[Option<i32>]) {
    let f = || v.iter().copied().collect::<Int32Chunked>();
    criterion::black_box(f());
}

fn bench_collect_optional_str(v: &[Option<String>]) {
    let f = || v.iter().map(|opt| opt.as_ref()).collect::<Utf8Chunked>();
    criterion::black_box(f());
}

fn create_array(size: i32, null_percentage: f32) -> Vec<Option<i32>> {
    let mut rng = StdRng::seed_from_u64(0);
    (0..size)
        .map(|i| {
            if rng.gen::<f32>() < null_percentage {
                None
            } else {
                Some(i)
            }
        })
        .collect()
}

fn create_string_array(
    size: i32,
    null_percentage: f32,
    lower_len: usize,
    upper_len: usize,
) -> Vec<Option<String>> {
    let mut rng = StdRng::seed_from_u64(0);
    (0..size)
        .map(|_| {
            if rng.gen::<f32>() < null_percentage {
                None
            } else {
                let length = rng.gen_range(lower_len..upper_len);
                let s: String = (0..length).map(|_| rng.gen::<u8>() as char).collect();
                Some(s)
            }
        })
        .collect()
}

fn add_benchmark(c: &mut Criterion) {
    let v = vec![true; 1024];
    c.bench_function("collect bool 1024", |b| b.iter(|| bench_collect_bool(&v)));
    let v = vec![true; 4096];
    c.bench_function("collect bool 4096", |b| b.iter(|| bench_collect_bool(&v)));

    let v = vec![1.0; 1024];
    c.bench_function("collect num 1024", |b| b.iter(|| bench_collect_num(&v)));
    let v = vec![1.0; 4096];
    c.bench_function("collect num 4096", |b| b.iter(|| bench_collect_num(&v)));

    let v = create_array(1024, 0.05);
    c.bench_function("collect optional_num 1024", |b| {
        b.iter(|| bench_collect_optional_num(&v))
    });
    let v = create_array(4096, 0.05);
    c.bench_function("collect optional_num 4096", |b| {
        b.iter(|| bench_collect_optional_num(&v))
    });

    let v = create_string_array(1024, 0.05, 5, 80);
    c.bench_function("collect optional_str 1024", |b| {
        b.iter(|| bench_collect_optional_str(&v))
    });
    let v = create_string_array(4096, 0.05, 5, 80);
    c.bench_function("collect optional_str 4096", |b| {
        b.iter(|| bench_collect_optional_str(&v))
    });
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
