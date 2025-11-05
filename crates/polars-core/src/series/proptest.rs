use std::ops::RangeInclusive;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::bitmap::bitmask::nth_set_bit_u32;
#[cfg(feature = "dtype-categorical")]
use polars_dtype::categorical::{CategoricalMapping, Categories, FrozenCategories};
use proptest::prelude::*;

#[cfg(feature = "dtype-categorical")]
use crate::chunked_array::builder::CategoricalChunkedBuilder;
use crate::prelude::{Int32Chunked, Int64Chunked, Int128Chunked, NamedFrom, Series, TimeUnit};
#[cfg(feature = "dtype-categorical")]
use crate::series::{Categorical8Type, DataType, IntoSeries};

// A global, thread-safe counter that will be used to ensure unique column names when the Series are created
// This is especially useful for when the Series strategies are combined to create a DataFrame strategy
static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn next_column_name() -> String {
    format!("col_{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SeriesArbitrarySelection: u32 {
        const BOOLEAN = 1;
        const UINT = 1 << 1;
        const INT = 1 << 2;
        const FLOAT = 1 << 3;
        const STRING = 1 << 4;
        const BINARY = 1 << 5;

        const TIME = 1 << 6;
        const DATETIME = 1 << 7;
        const DATE = 1 << 8;
        const DURATION = 1 << 9;
        const DECIMAL = 1 << 10;
        const CATEGORICAL = 1 << 11;
        const ENUM = 1 << 12;
    }
}

impl SeriesArbitrarySelection {
    pub fn physical() -> Self {
        Self::BOOLEAN | Self::UINT | Self::INT | Self::FLOAT | Self::STRING | Self::BINARY
    }

    pub fn logical() -> Self {
        Self::TIME
            | Self::DATETIME
            | Self::DATE
            | Self::DURATION
            | Self::DECIMAL
            | Self::CATEGORICAL
            | Self::ENUM
    }
}

#[derive(Clone)]
pub struct SeriesArbitraryOptions {
    pub allowed_dtypes: SeriesArbitrarySelection,
    pub series_length_range: RangeInclusive<usize>,
    pub categories_range: RangeInclusive<usize>,
}

impl Default for SeriesArbitraryOptions {
    fn default() -> Self {
        Self {
            allowed_dtypes: SeriesArbitrarySelection::all(),
            series_length_range: 0..=5,
            categories_range: 0..=3,
        }
    }
}

pub fn series_strategy(options: Rc<SeriesArbitraryOptions>) -> impl Strategy<Value = Series> {
    use SeriesArbitrarySelection as S;
    #[allow(unused_mut)]
    let mut allowed_dtypes = options.allowed_dtypes;

    let num_possible_types = allowed_dtypes.bits().count_ones();
    assert!(num_possible_types > 0);

    (0..num_possible_types).prop_flat_map(move |i| {
        let selection =
            S::from_bits_retain(1 << nth_set_bit_u32(options.allowed_dtypes.bits(), i).unwrap());

        match selection {
            _ if selection == S::BOOLEAN => {
                series_boolean_strategy(options.series_length_range.clone()).boxed()
            },
            _ if selection == S::UINT => {
                series_uint_strategy(options.series_length_range.clone()).boxed()
            },
            _ if selection == S::INT => {
                series_int_strategy(options.series_length_range.clone()).boxed()
            },
            _ if selection == S::FLOAT => {
                series_float_strategy(options.series_length_range.clone()).boxed()
            },
            _ if selection == S::STRING => {
                series_string_strategy(options.series_length_range.clone()).boxed()
            },
            _ if selection == S::BINARY => {
                series_binary_strategy(options.series_length_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-time")]
            _ if selection == S::TIME => {
                series_time_strategy(options.series_length_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-datetime")]
            _ if selection == S::DATETIME => {
                series_datetime_strategy(options.series_length_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-date")]
            _ if selection == S::DATE => {
                series_date_strategy(options.series_length_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-duration")]
            _ if selection == S::DURATION => {
                series_duration_strategy(options.series_length_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-decimal")]
            _ if selection == S::DECIMAL => {
                series_decimal_strategy(options.series_length_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-categorical")]
            _ if selection == S::CATEGORICAL => series_categorical_strategy(
                options.series_length_range.clone(),
                options.categories_range.clone(),
            )
            .boxed(),
            #[cfg(feature = "dtype-categorical")]
            _ if selection == S::ENUM => series_enum_strategy(
                options.series_length_range.clone(),
                options.categories_range.clone(),
            )
            .boxed(),
            _ => unreachable!(),
        }
    })
}

fn series_boolean_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop::collection::vec(any::<bool>(), series_length_range)
        .prop_map(|bools| Series::new(next_column_name().into(), bools))
}

fn series_uint_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop_oneof![
        prop::collection::vec(any::<u8>(), series_length_range.clone())
            .prop_map(|uints| Series::new(next_column_name().into(), uints)),
        prop::collection::vec(any::<u16>(), series_length_range.clone())
            .prop_map(|uints| Series::new(next_column_name().into(), uints)),
        prop::collection::vec(any::<u32>(), series_length_range.clone())
            .prop_map(|uints| Series::new(next_column_name().into(), uints)),
        prop::collection::vec(any::<u64>(), series_length_range.clone())
            .prop_map(|uints| Series::new(next_column_name().into(), uints)),
        prop::collection::vec(any::<u128>(), series_length_range)
            .prop_map(|uints| Series::new(next_column_name().into(), uints)),
    ]
}

fn series_int_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop_oneof![
        prop::collection::vec(any::<i8>(), series_length_range.clone())
            .prop_map(|ints| Series::new(next_column_name().into(), ints)),
        prop::collection::vec(any::<i16>(), series_length_range.clone())
            .prop_map(|ints| Series::new(next_column_name().into(), ints)),
        prop::collection::vec(any::<i32>(), series_length_range.clone())
            .prop_map(|ints| Series::new(next_column_name().into(), ints)),
        prop::collection::vec(any::<i64>(), series_length_range.clone())
            .prop_map(|ints| Series::new(next_column_name().into(), ints)),
        prop::collection::vec(any::<i128>(), series_length_range)
            .prop_map(|ints| Series::new(next_column_name().into(), ints)),
    ]
}

fn series_float_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop_oneof![
        prop::collection::vec(any::<f32>(), series_length_range.clone())
            .prop_map(|floats| Series::new(next_column_name().into(), floats)),
        prop::collection::vec(any::<f64>(), series_length_range)
            .prop_map(|floats| Series::new(next_column_name().into(), floats)),
    ]
}

fn series_string_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop::collection::vec(any::<String>(), series_length_range)
        .prop_map(|strings| Series::new(next_column_name().into(), strings))
}

fn series_binary_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop::collection::vec(any::<u8>(), series_length_range)
        .prop_map(|binaries| Series::new(next_column_name().into(), binaries))
}

#[cfg(feature = "dtype-time")]
fn series_time_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop::collection::vec(
        0i64..86_400_000_000_000i64, // Time range: 0 to just under 24 hours in nanoseconds
        series_length_range,
    )
    .prop_map(|times| {
        Int64Chunked::new(next_column_name().into(), &times)
            .into_time()
            .into_series()
    })
}

#[cfg(feature = "dtype-datetime")]
fn series_datetime_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop::collection::vec(
        0i64..i64::MAX, // Datetime range: 0 (1970-01-01) to i64::MAX in milliseconds since UNIX epoch
        series_length_range,
    )
    .prop_map(|datetimes| {
        Int64Chunked::new(next_column_name().into(), &datetimes)
            .into_datetime(TimeUnit::Milliseconds, None)
            .into_series()
    })
}

#[cfg(feature = "dtype-date")]
fn series_date_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop::collection::vec(
        0i32..50_000i32, // Date range: 0 (1970-01-01) to ~50,000 days (~137 years, roughly 1970-2107)
        series_length_range,
    )
    .prop_map(|dates| {
        Int32Chunked::new(next_column_name().into(), &dates)
            .into_date()
            .into_series()
    })
}

#[cfg(feature = "dtype-duration")]
fn series_duration_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop::collection::vec(
        i64::MIN..i64::MAX, // Duration range: full i64 range in milliseconds (can be negative for time differences)
        series_length_range,
    )
    .prop_map(|durations| {
        Int64Chunked::new(next_column_name().into(), &durations)
            .into_duration(TimeUnit::Milliseconds)
            .into_series()
    })
}

#[cfg(feature = "dtype-decimal")]
fn series_decimal_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop::collection::vec(i128::MIN..i128::MAX, series_length_range).prop_map(|decimals| {
        Int128Chunked::new(next_column_name().into(), &decimals)
            .into_decimal_unchecked(38, 9) // precision = 38 (max for i128), scale = 9 (9 decimal places)
            .into_series()
    })
}

#[cfg(feature = "dtype-categorical")]
fn series_categorical_strategy(
    series_length_range: RangeInclusive<usize>,
    categories_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    categories_range
        .prop_flat_map(move |n_categories| {
            let possible_categories: Vec<String> =
                (0..n_categories).map(|i| format!("category{i}")).collect();

            prop::collection::vec(
                prop::sample::select(possible_categories),
                series_length_range.clone(),
            )
        })
        .prop_map(|categories| {
            // Using Categorical8Type (u8 backing) which supports up to 256 unique categories
            let mut builder = CategoricalChunkedBuilder::<Categorical8Type>::new(
                next_column_name().into(),
                DataType::Categorical(Categories::global(), Arc::new(CategoricalMapping::new(256))),
            );

            for category in categories {
                builder.append_str(&category).unwrap();
            }

            builder.finish().into_series()
        })
}

#[cfg(feature = "dtype-categorical")]
fn series_enum_strategy(
    series_length_range: RangeInclusive<usize>,
    categories_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    categories_range
        .prop_flat_map(move |n_categories| {
            let possible_categories: Vec<String> =
                (0..n_categories).map(|i| format!("category{i}")).collect();

            (
                Just(possible_categories.clone()),
                prop::collection::vec(
                    prop::sample::select(possible_categories),
                    series_length_range.clone(),
                ),
            )
        })
        .prop_map(|(possible_categories, sampled_categories)| {
            let frozen_categories =
                FrozenCategories::new(possible_categories.iter().map(|s| s.as_str())).unwrap();
            let mapping = frozen_categories.mapping().clone();

            // Using Categorical8Type (u8 backing) which supports up to 256 unique categories
            let mut builder = CategoricalChunkedBuilder::<Categorical8Type>::new(
                next_column_name().into(),
                DataType::Enum(frozen_categories, mapping),
            );

            for category in sampled_categories {
                builder.append_str(&category).unwrap();
            }

            builder.finish().into_series()
        })
}
