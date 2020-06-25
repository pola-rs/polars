// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#[macro_use]
extern crate criterion;
use criterion::Criterion;
use rand::distributions::{Distribution, Standard};
use rand::prelude::random;
use rand::Rng;

use std::sync::Arc;

extern crate arrow;

use arrow::array::*;
use arrow::compute::{cast, take};
use arrow::datatypes::*;

// cast array from specified primitive array type to desired data type
fn create_numeric<T>(size: usize) -> ArrayRef
where
    T: ArrowNumericType,
    Standard: Distribution<T::Native>,
    PrimitiveArray<T>: std::convert::From<Vec<T::Native>>,
{
    Arc::new(PrimitiveArray::<T>::from(vec![random::<T::Native>(); size])) as ArrayRef
}

fn create_random_index(size: usize) -> UInt32Array {
    let mut rng = rand::thread_rng();
    let ints = Int32Array::from(vec![rng.gen_range(-24i32, size as i32); size]);
    // cast to u32, conveniently marking negative values as nulls
    UInt32Array::from(
        cast(&(Arc::new(ints) as ArrayRef), &DataType::UInt32)
            .unwrap()
            .data(),
    )
}

fn take_numeric<T>(size: usize, index_len: usize) -> ()
where
    T: ArrowNumericType,
    Standard: Distribution<T::Native>,
    PrimitiveArray<T>: std::convert::From<Vec<T::Native>>,
    T::Native: num::NumCast,
{
    let array = create_numeric::<T>(size);
    let index = create_random_index(index_len);
    criterion::black_box(take(&array, &index, None).unwrap());
}

fn take_boolean(size: usize, index_len: usize) -> () {
    let array = Arc::new(BooleanArray::from(vec![random::<bool>(); size])) as ArrayRef;
    let index = create_random_index(index_len);
    criterion::black_box(take(&array, &index, None).unwrap());
}

fn add_benchmark(c: &mut Criterion) {
    c.bench_function("take u8 256", |b| {
        b.iter(|| take_numeric::<UInt8Type>(256, 256))
    });
    c.bench_function("take u8 512", |b| {
        b.iter(|| take_numeric::<UInt8Type>(512, 512))
    });
    c.bench_function("take u8 1024", |b| {
        b.iter(|| take_numeric::<UInt8Type>(1024, 1024))
    });
    c.bench_function("take i32 256", |b| {
        b.iter(|| take_numeric::<Int32Type>(256, 256))
    });
    c.bench_function("take i32 512", |b| {
        b.iter(|| take_numeric::<Int32Type>(512, 512))
    });
    c.bench_function("take i32 1024", |b| {
        b.iter(|| take_numeric::<Int32Type>(1024, 1024))
    });
    c.bench_function("take bool 256", |b| b.iter(|| take_boolean(256, 256)));
    c.bench_function("take bool 512", |b| b.iter(|| take_boolean(512, 512)));
    c.bench_function("take bool 1024", |b| b.iter(|| take_boolean(1024, 1024)));
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
