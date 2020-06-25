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

extern crate arrow;

use arrow::array::*;
use arrow::compute::*;
use arrow::datatypes::ArrowNumericType;

fn create_array(size: usize) -> Float32Array {
    let mut builder = Float32Builder::new(size);
    for i in 0..size {
        if i % 2 == 0 {
            builder.append_value(1.0).unwrap();
        } else {
            builder.append_value(0.0).unwrap();
        }
    }
    builder.finish()
}

fn bench_eq<T>(arr_a: &PrimitiveArray<T>, arr_b: &PrimitiveArray<T>)
where
    T: ArrowNumericType,
{
    eq(criterion::black_box(arr_a), criterion::black_box(arr_b)).unwrap();
}

fn bench_eq_scalar<T>(arr_a: &PrimitiveArray<T>, value_b: T::Native)
where
    T: ArrowNumericType,
{
    eq_scalar(criterion::black_box(arr_a), criterion::black_box(value_b)).unwrap();
}

fn bench_neq<T>(arr_a: &PrimitiveArray<T>, arr_b: &PrimitiveArray<T>)
where
    T: ArrowNumericType,
{
    neq(criterion::black_box(arr_a), criterion::black_box(arr_b)).unwrap();
}

fn bench_neq_scalar<T>(arr_a: &PrimitiveArray<T>, value_b: T::Native)
where
    T: ArrowNumericType,
{
    neq_scalar(criterion::black_box(arr_a), criterion::black_box(value_b)).unwrap();
}

fn bench_lt<T>(arr_a: &PrimitiveArray<T>, arr_b: &PrimitiveArray<T>)
where
    T: ArrowNumericType,
{
    lt(criterion::black_box(arr_a), criterion::black_box(arr_b)).unwrap();
}

fn bench_lt_scalar<T>(arr_a: &PrimitiveArray<T>, value_b: T::Native)
where
    T: ArrowNumericType,
{
    lt_scalar(criterion::black_box(arr_a), criterion::black_box(value_b)).unwrap();
}

fn bench_lt_eq<T>(arr_a: &PrimitiveArray<T>, arr_b: &PrimitiveArray<T>)
where
    T: ArrowNumericType,
{
    lt_eq(criterion::black_box(arr_a), criterion::black_box(arr_b)).unwrap();
}

fn bench_lt_eq_scalar<T>(arr_a: &PrimitiveArray<T>, value_b: T::Native)
where
    T: ArrowNumericType,
{
    lt_eq_scalar(criterion::black_box(arr_a), criterion::black_box(value_b)).unwrap();
}

fn bench_gt<T>(arr_a: &PrimitiveArray<T>, arr_b: &PrimitiveArray<T>)
where
    T: ArrowNumericType,
{
    gt(criterion::black_box(arr_a), criterion::black_box(arr_b)).unwrap();
}

fn bench_gt_scalar<T>(arr_a: &PrimitiveArray<T>, value_b: T::Native)
where
    T: ArrowNumericType,
{
    gt_scalar(criterion::black_box(arr_a), criterion::black_box(value_b)).unwrap();
}

fn bench_gt_eq<T>(arr_a: &PrimitiveArray<T>, arr_b: &PrimitiveArray<T>)
where
    T: ArrowNumericType,
{
    gt_eq(criterion::black_box(arr_a), criterion::black_box(arr_b)).unwrap();
}

fn bench_gt_eq_scalar<T>(arr_a: &PrimitiveArray<T>, value_b: T::Native)
where
    T: ArrowNumericType,
{
    gt_eq_scalar(criterion::black_box(arr_a), criterion::black_box(value_b)).unwrap();
}

fn add_benchmark(c: &mut Criterion) {
    let size = 65536;
    let arr_a = create_array(size);
    let arr_b = create_array(size);

    c.bench_function("eq Float32", |b| b.iter(|| bench_eq(&arr_a, &arr_b)));
    c.bench_function("eq scalar Float32", |b| {
        b.iter(|| bench_eq_scalar(&arr_a, 1.0))
    });

    c.bench_function("neq Float32", |b| b.iter(|| bench_neq(&arr_a, &arr_b)));
    c.bench_function("neq scalar Float32", |b| {
        b.iter(|| bench_neq_scalar(&arr_a, 1.0))
    });

    c.bench_function("lt Float32", |b| b.iter(|| bench_lt(&arr_a, &arr_b)));
    c.bench_function("lt scalar Float32", |b| {
        b.iter(|| bench_lt_scalar(&arr_a, 1.0))
    });

    c.bench_function("lt_eq Float32", |b| b.iter(|| bench_lt_eq(&arr_a, &arr_b)));
    c.bench_function("lt_eq scalar Float32", |b| {
        b.iter(|| bench_lt_eq_scalar(&arr_a, 1.0))
    });

    c.bench_function("gt Float32", |b| b.iter(|| bench_gt(&arr_a, &arr_b)));
    c.bench_function("gt scalar Float32", |b| {
        b.iter(|| bench_gt_scalar(&arr_a, 1.0))
    });

    c.bench_function("gt_eq Float32", |b| b.iter(|| bench_gt_eq(&arr_a, &arr_b)));
    c.bench_function("gt_eq scalar Float32", |b| {
        b.iter(|| bench_gt_eq_scalar(&arr_a, 1.0))
    });
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
