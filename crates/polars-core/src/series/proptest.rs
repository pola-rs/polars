use std::ops::RangeInclusive;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::bitmap::bitmask::nth_set_bit_u32;
use proptest::prelude::*;

use crate::prelude::{NamedFrom, Series};

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
    }
}

impl SeriesArbitrarySelection {
    pub fn physical() -> Self {
        Self::BOOLEAN | Self::UINT | Self::INT | Self::FLOAT | Self::STRING | Self::BINARY
    }
}

#[derive(Clone)]
pub struct SeriesArbitraryOptions {
    pub allowed_dtypes: SeriesArbitrarySelection,
    pub series_length_range: RangeInclusive<usize>,
}

impl Default for SeriesArbitraryOptions {
    fn default() -> Self {
        Self {
            allowed_dtypes: SeriesArbitrarySelection::all(),
            series_length_range: 0..=5,
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
            _ => unreachable!(),
        }
    })
}

pub fn series_boolean_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop::collection::vec(any::<bool>(), series_length_range)
        .prop_map(|bools| Series::new(next_column_name().into(), bools))
}

pub fn series_uint_strategy(
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

pub fn series_int_strategy(
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

pub fn series_float_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop_oneof![
        prop::collection::vec(any::<f32>(), series_length_range.clone())
            .prop_map(|floats| Series::new(next_column_name().into(), floats)),
        prop::collection::vec(any::<f64>(), series_length_range)
            .prop_map(|floats| Series::new(next_column_name().into(), floats)),
    ]
}

pub fn series_string_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop::collection::vec(any::<String>(), series_length_range)
        .prop_map(|strings| Series::new(next_column_name().into(), strings))
}

pub fn series_binary_strategy(
    series_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = Series> {
    prop::collection::vec(any::<u8>(), series_length_range)
        .prop_map(|binaries| Series::new(next_column_name().into(), binaries))
}
