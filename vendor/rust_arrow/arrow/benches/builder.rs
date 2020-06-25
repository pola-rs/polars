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

extern crate arrow;
extern crate criterion;
extern crate rand;

use std::mem::size_of;

use criterion::*;
use rand::distributions::Standard;
use rand::{thread_rng, Rng};

use arrow::array::*;

// Build arrays with 512k elements.
const BATCH_SIZE: usize = 8 << 10;
const NUM_BATCHES: usize = 64;

fn bench_primitive(c: &mut Criterion) {
    let data: [i64; BATCH_SIZE] = [100; BATCH_SIZE];
    c.bench(
        "bench_primitive",
        Benchmark::new("bench_primitive", move |b| {
            b.iter(|| {
                let mut builder = Int64Builder::new(64);
                for _ in 0..NUM_BATCHES {
                    let _ = black_box(builder.append_slice(&data[..]));
                }
                black_box(builder.finish());
            })
        })
        .throughput(Throughput::Bytes(
            ((data.len() * NUM_BATCHES * size_of::<i64>()) as u32).into(),
        )),
    );
}

fn bench_bool(c: &mut Criterion) {
    let data: Vec<bool> = thread_rng()
        .sample_iter(&Standard)
        .take(BATCH_SIZE)
        .collect();
    let data_len = data.len();
    c.bench(
        "bench_bool",
        Benchmark::new("bench_bool", move |b| {
            b.iter(|| {
                let mut builder = BooleanBuilder::new(64);
                for _ in 0..NUM_BATCHES {
                    let _ = black_box(builder.append_slice(&data[..]));
                }
                black_box(builder.finish());
            })
        })
        .throughput(Throughput::Bytes(
            ((data_len * NUM_BATCHES * size_of::<bool>()) as u32).into(),
        )),
    );
}

criterion_group!(benches, bench_primitive, bench_bool);
criterion_main!(benches);
