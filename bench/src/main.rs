#![feature(test)]
extern crate test;
use arrow::array::{Array, ArrayRef};
use arrow::{
    array::{PrimitiveArray, PrimitiveBuilder},
    datatypes::Int32Type,
};
use polars::prelude::*;
use std::sync::Arc;
use test::Bencher;

const SIZE: usize = 1000;
const N: usize = 10;

fn create_arrow_array(size: usize) -> Arc<dyn Array> {
    let mut builder = PrimitiveBuilder::<Int32Type>::new(size);
    for i in 0..size {
        builder.append_value(i as i32).expect("append");
    }
    Arc::new(builder.finish())
}

fn n_arrow_arrays(n: usize, size: usize) -> Vec<Arc<dyn Array>> {
    let mut arrays = Vec::with_capacity(n);
    for _ in 0..n {
        arrays.push(create_arrow_array(size))
    }
    arrays
}

#[bench]
fn bench_vec_clone(b: &mut Bencher) {
    let arrays = n_arrow_arrays(N, SIZE);
    b.iter(|| arrays.clone())
}

#[bench]
fn bench_arc_clone(b: &mut Bencher) {
    let arrays = n_arrow_arrays(N, SIZE);
    let arrays = Arc::new(arrays);
    b.iter(|| arrays.clone())
}

#[bench]
fn bench_series_clone(b: &mut Bencher) {
    let arrays = n_arrow_arrays(N, SIZE);

    let ca = ChunkedArray::new_from_chunks("a", arrays);
    let s = Series::Int32(ca);
    b.iter(|| s.clone())
}

#[bench]
fn bench_data_clone(b: &mut Bencher) {
    // Is not really comparable, as arrow arrays also have null bits
    let mut arrays = Vec::with_capacity(N);
    for _ in 0..N {
        arrays.push((0..SIZE).collect::<Vec<_>>());
    }

    b.iter(|| arrays.clone())
}
fn main() {}
