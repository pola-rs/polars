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

use std::sync::Arc;

extern crate arrow;

use arrow::array::*;
use arrow::compute::cast;
use arrow::datatypes::*;

// cast array from specified primitive array type to desired data type
fn cast_array<FROM>(size: usize, to_type: DataType) -> ()
where
    FROM: ArrowNumericType,
    Standard: Distribution<FROM::Native>,
    PrimitiveArray<FROM>: std::convert::From<Vec<FROM::Native>>,
{
    let array = Arc::new(PrimitiveArray::<FROM>::from(vec![
        random::<FROM::Native>();
        size
    ])) as ArrayRef;
    criterion::black_box(cast(&array, &to_type).unwrap());
}

// cast timestamp array from specified primitive array type to desired data type
fn cast_timestamp_array<FROM>(size: usize, to_type: DataType) -> ()
where
    FROM: ArrowTimestampType,
    Standard: Distribution<i64>,
{
    let array = Arc::new(PrimitiveArray::<FROM>::from_vec(
        vec![random::<i64>(); size],
        None,
    )) as ArrayRef;
    criterion::black_box(cast(&array, &to_type).unwrap());
}

fn add_benchmark(c: &mut Criterion) {
    c.bench_function("cast int32 to int32 512", |b| {
        b.iter(|| cast_array::<Int32Type>(512, DataType::Int32))
    });
    c.bench_function("cast int32 to uint32 512", |b| {
        b.iter(|| cast_array::<Int32Type>(512, DataType::UInt32))
    });
    c.bench_function("cast int32 to float32 512", |b| {
        b.iter(|| cast_array::<Int32Type>(512, DataType::Float32))
    });
    c.bench_function("cast int32 to float64 512", |b| {
        b.iter(|| cast_array::<Int32Type>(512, DataType::Float64))
    });
    c.bench_function("cast int32 to int64 512", |b| {
        b.iter(|| cast_array::<Int32Type>(512, DataType::Int64))
    });
    c.bench_function("cast float32 to int32 512", |b| {
        b.iter(|| cast_array::<Float32Type>(512, DataType::Int32))
    });
    c.bench_function("cast float64 to float32 512", |b| {
        b.iter(|| cast_array::<Float64Type>(512, DataType::Float32))
    });
    c.bench_function("cast float64 to uint64 512", |b| {
        b.iter(|| cast_array::<Float64Type>(512, DataType::UInt64))
    });
    c.bench_function("cast int64 to int32 512", |b| {
        b.iter(|| cast_array::<Int64Type>(512, DataType::Int32))
    });
    c.bench_function("cast date64 to date32 512", |b| {
        b.iter(|| cast_array::<Date64Type>(512, DataType::Date32(DateUnit::Day)))
    });
    c.bench_function("cast date32 to date64 512", |b| {
        b.iter(|| cast_array::<Date32Type>(512, DataType::Date64(DateUnit::Millisecond)))
    });
    c.bench_function("cast time32s to time32ms 512", |b| {
        b.iter(|| {
            cast_array::<Time32SecondType>(512, DataType::Time32(TimeUnit::Millisecond))
        })
    });
    c.bench_function("cast time32s to time64us 512", |b| {
        b.iter(|| {
            cast_array::<Time32SecondType>(512, DataType::Time64(TimeUnit::Microsecond))
        })
    });
    c.bench_function("cast time64ns to time32s 512", |b| {
        b.iter(|| {
            cast_array::<Time64NanosecondType>(512, DataType::Time32(TimeUnit::Second))
        })
    });
    c.bench_function("cast timestamp_ns to timestamp_s 512", |b| {
        b.iter(|| {
            cast_timestamp_array::<TimestampNanosecondType>(
                512,
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            )
        })
    });
    c.bench_function("cast timestamp_ms to timestamp_ns 512", |b| {
        b.iter(|| {
            cast_timestamp_array::<TimestampMillisecondType>(
                512,
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            )
        })
    });
    c.bench_function("cast timestamp_ms to i64 512", |b| {
        b.iter(|| cast_timestamp_array::<TimestampMillisecondType>(512, DataType::Int64))
    });
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
