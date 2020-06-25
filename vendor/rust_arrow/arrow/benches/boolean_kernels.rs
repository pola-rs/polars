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
use arrow::compute::kernels::boolean as boolean_kernels;

///  Helper function to create arrays
fn create_boolean_array(size: usize) -> BooleanArray {
    let mut builder = BooleanBuilder::new(size);
    for i in 0..size {
        if i % 2 == 0 {
            builder.append_value(true).unwrap();
        } else {
            builder.append_value(false).unwrap();
        }
    }
    builder.finish()
}

/// Benchmark for `AND`
fn bench_and(size: usize) {
    let buffer_a = create_boolean_array(size);
    let buffer_b = create_boolean_array(size);
    criterion::black_box(boolean_kernels::and(&buffer_a, &buffer_b).unwrap());
}

/// Benchmark for `OR`
fn bench_or(size: usize) {
    let buffer_a = create_boolean_array(size);
    let buffer_b = create_boolean_array(size);
    criterion::black_box(boolean_kernels::or(&buffer_a, &buffer_b).unwrap());
}

/// Benchmark for `NOT`
fn bench_not(size: usize) {
    let buffer = create_boolean_array(size);
    criterion::black_box(boolean_kernels::not(&buffer).unwrap());
}

fn add_benchmark(c: &mut Criterion) {
    c.bench_function("and", |b| b.iter(|| bench_and(512)));
    c.bench_function("or", |b| b.iter(|| bench_or(512)));
    c.bench_function("not", |b| b.iter(|| bench_not(512)));
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
