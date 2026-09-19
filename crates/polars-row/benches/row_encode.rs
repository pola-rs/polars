use std::hint::black_box;
use std::time::Instant;

use polars_arrow::array::{BinaryArray, PrimitiveArray, Utf8ViewArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use polars_row::decode::decode_rows_from_binary;
use polars_row::{ArrayRef, RowEncodingOptions, convert_columns};

const N: usize = 1 << 20;

fn rng(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn u64_col(seed: &mut u64) -> ArrayRef {
    PrimitiveArray::<u64>::from_vec((0..N).map(|_| rng(seed)).collect()).boxed()
}
fn i64_col(seed: &mut u64) -> ArrayRef {
    PrimitiveArray::<i64>::from_vec((0..N).map(|_| rng(seed) as i64).collect()).boxed()
}
fn i32_col(seed: &mut u64) -> ArrayRef {
    PrimitiveArray::<i32>::from_vec((0..N).map(|_| rng(seed) as i32).collect()).boxed()
}
fn f64_col(seed: &mut u64) -> ArrayRef {
    PrimitiveArray::<f64>::from_vec((0..N).map(|_| rng(seed) as f64 / 7.0).collect()).boxed()
}
fn u64_null_col(seed: &mut u64) -> ArrayRef {
    let v = PrimitiveArray::<u64>::from_vec((0..N).map(|_| rng(seed)).collect());
    let validity: Bitmap = (0..N).map(|_| !rng(seed).is_multiple_of(10)).collect();
    v.with_validity(Some(validity)).boxed()
}
fn str_col(seed: &mut u64, min: usize, max: usize) -> ArrayRef {
    let strs: Vec<String> = (0..N)
        .map(|_| {
            let len = min + (rng(seed) as usize % (max - min + 1));
            (0..len)
                .map(|_| (b'a' + (rng(seed) % 26) as u8) as char)
                .collect()
        })
        .collect();
    Utf8ViewArray::from_slice_values(&strs).boxed()
}

fn bench_encode(name: &str, cols: &[ArrayRef], opts: &[RowEncodingOptions], iters: usize) {
    let dicts = vec![None; cols.len()];
    // warmup
    let rows = convert_columns(N, cols, opts, &dicts);
    let bytes = rows.into_array().values().len();
    let mut best = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let r = convert_columns(N, black_box(cols), opts, &dicts);
        black_box(&r);
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!(
        "encode {name:<28} {:>8.2} ms  {:>7.2} ns/row  {:>6.2} GB/s",
        best * 1e3,
        best * 1e9 / N as f64,
        bytes as f64 / best / 1e9
    );
}

fn bench_decode(name: &str, cols: &[ArrayRef], opts: &[RowEncodingOptions], iters: usize) {
    let dicts = vec![None; cols.len()];
    let rows = convert_columns(N, cols, opts, &dicts);
    let arr: BinaryArray<i64> = rows.into_array();
    let dtypes: Vec<ArrowDataType> = cols.iter().map(|c| c.dtype().clone()).collect();
    let mut buf = Vec::new();
    let mut best = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let r =
            unsafe { decode_rows_from_binary(black_box(&arr), opts, &dicts, &dtypes, &mut buf) };
        black_box(&r);
        best = best.min(t.elapsed().as_secs_f64());
    }
    let mut best_rows = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        buf.clear();
        buf.extend(black_box(&arr).values_iter());
        black_box(&buf);
        best_rows = best_rows.min(t.elapsed().as_secs_f64());
    }
    println!(
        "decode {name:<28} {:>8.2} ms  {:>7.2} ns/row  (rows setup {:>5.2} ns/row)",
        best * 1e3,
        best * 1e9 / N as f64,
        best_rows * 1e9 / N as f64,
    );
}

fn main() {
    let mut seed = 0x1234_5678_9abc_def1u64;
    let sorted = RowEncodingOptions::new_sorted(false, false);
    let desc = RowEncodingOptions::new_sorted(true, false);
    let unsorted = RowEncodingOptions::new_unsorted();

    let u64c = u64_col(&mut seed);
    let i64c = i64_col(&mut seed);
    let i32c = i32_col(&mut seed);
    let f64c = f64_col(&mut seed);
    let u64n = u64_null_col(&mut seed);
    let strs = str_col(&mut seed, 4, 12);
    let strl = str_col(&mut seed, 20, 60);

    let cases: Vec<(&str, Vec<ArrayRef>, Vec<RowEncodingOptions>)> = vec![
        ("u64", vec![u64c.clone()], vec![sorted]),
        ("u64 desc", vec![u64c.clone()], vec![desc]),
        ("u64 unsorted", vec![u64c.clone()], vec![unsorted]),
        ("u64 10% null", vec![u64n.clone()], vec![sorted]),
        ("i32", vec![i32c.clone()], vec![sorted]),
        ("f64", vec![f64c.clone()], vec![sorted]),
        (
            "i64,f64,i32,u64",
            vec![i64c.clone(), f64c.clone(), i32c.clone(), u64c.clone()],
            vec![sorted; 4],
        ),
        (
            "i64,f64,i32,u64 unsorted",
            vec![i64c.clone(), f64c.clone(), i32c.clone(), u64c.clone()],
            vec![unsorted; 4],
        ),
        ("str short", vec![strs.clone()], vec![sorted]),
        ("str short unsorted", vec![strs.clone()], vec![unsorted]),
        ("str long", vec![strl.clone()], vec![sorted]),
        ("str long unsorted", vec![strl.clone()], vec![unsorted]),
        (
            "str short,u64",
            vec![strs.clone(), u64c.clone()],
            vec![sorted; 2],
        ),
        (
            "str short,u64 unsorted",
            vec![strs.clone(), u64c.clone()],
            vec![unsorted; 2],
        ),
    ];

    let filter = std::env::var("ROW_BENCH_FILTER").unwrap_or_default();
    let iters: usize = std::env::var("ROW_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    for (name, cols, opts) in &cases {
        if filter.is_empty() || format!("encode {name}").contains(&filter) {
            bench_encode(name, cols, opts, iters);
        }
    }
    println!();
    for (name, cols, opts) in &cases {
        if filter.is_empty() || format!("decode {name}").contains(&filter) {
            bench_decode(name, cols, opts, iters);
        }
    }
}
