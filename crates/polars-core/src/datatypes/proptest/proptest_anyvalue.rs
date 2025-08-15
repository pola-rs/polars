use std::ops::RangeInclusive;
use std::rc::Rc;
use std::sync::Arc;

use arrow::bitmap::bitmask::nth_set_bit_u32;
use polars_utils::pl_str::PlSmallStr;
use proptest::prelude::*;

use super::super::{AnyValue, CategoricalMapping, OwnedObject, PolarsObjectSafe, TimeUnit};
use crate::prelude::PolarsObject;

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct AnyValueArbitrarySelection: u32 {
        const NULL              = 1;
        const BOOLEAN           = 1 << 1;
        const STRING            = 1 << 2;
        const STRING_OWNED      = 1 << 3;
        const UINT8             = 1 << 4;
        const UINT16            = 1 << 5;
        const UINT32            = 1 << 6;
        const UINT64            = 1 << 7;
        const INT8              = 1 << 8;
        const INT16             = 1 << 9;
        const INT32             = 1 << 10;
        const INT64             = 1 << 11;
        const INT128            = 1 << 12;
        const FLOAT32           = 1 << 13;
        const FLOAT64           = 1 << 14;
        const DATE              = 1 << 15;
        const TIME              = 1 << 16;
        const BINARY            = 1 << 17;
        const BINARY_OWNED      = 1 << 18;
        const OBJECT            = 1 << 19;
        const OBJECT_OWNED      = 1 << 20;

        const DATETIME          = 1 << 21;
        const DATETIME_OWNED    = 1 << 22;
        const DURATION          = 1 << 23;
        const DECIMAL           = 1 << 24;
        const CATEGORICAL       = 1 << 25;
        const CATEGORICAL_OWNED = 1 << 26;
        const ENUM              = 1 << 27;
        const ENUM_OWNED        = 1 << 28;

        const LIST              = 1 << 29;
        const ARRAY             = 1 << 30;
        const STRUCT            = 1 << 31;
        // const STRUCT_OWNED      = 1 << 32;
    }
}

impl AnyValueArbitrarySelection {
    pub fn nested() -> Self {
        Self::LIST | Self::ARRAY | Self::STRUCT
        // | Self::STRUCT_OWNED
    }
}

#[derive(Clone)]
pub struct AnyValueArbitraryOptions {
    pub allowed_dtypes: AnyValueArbitrarySelection,
    pub vector_length_range: RangeInclusive<usize>,
    pub decimal_precision_range: RangeInclusive<usize>,
    pub categories_range: RangeInclusive<usize>,
    // TODO: Add fields later
}

impl Default for AnyValueArbitraryOptions {
    fn default() -> Self {
        Self {
            allowed_dtypes: AnyValueArbitrarySelection::all(),
            vector_length_range: 0..=100,
            decimal_precision_range: 1..=38,
            categories_range: 0..=3, // TODO: Add fields later
        }
    }
}

pub fn anyvalue_strategy(
    options: Rc<AnyValueArbitraryOptions>,
    include_nested: bool,
    // TODO: Other input parameters
) -> impl Strategy<Value = AnyValue<'static>> {
    use AnyValueArbitrarySelection as S;
    let mut allowed_dtypes = options.allowed_dtypes;

    if !include_nested {
        allowed_dtypes &= !S::nested();
    }

    let num_possible_types = allowed_dtypes.bits().count_ones();
    assert!(num_possible_types > 0);

    (0..num_possible_types).prop_flat_map(move |i| {
        let selection =
            S::from_bits_retain(1 << nth_set_bit_u32(allowed_dtypes.bits(), i).unwrap());

        match selection {
            _ if selection == S::NULL => Just(AnyValue::Null).boxed(),
            _ if selection == S::BOOLEAN => any::<bool>().prop_map(AnyValue::Boolean).boxed(),
            _ if selection == S::STRING => ".*"
                .prop_map(|s| AnyValue::String(Box::leak(s.into_boxed_str())))
                .boxed(),
            _ if selection == S::STRING_OWNED => any::<String>()
                .prop_map(|s| AnyValue::StringOwned(PlSmallStr::from_string(s)))
                .boxed(),
            _ if selection == S::UINT8 => any::<u8>().prop_map(AnyValue::UInt8).boxed(),
            _ if selection == S::UINT16 => any::<u16>().prop_map(AnyValue::UInt16).boxed(),
            _ if selection == S::UINT32 => any::<u32>().prop_map(AnyValue::UInt32).boxed(),
            _ if selection == S::UINT64 => any::<u64>().prop_map(AnyValue::UInt64).boxed(),
            _ if selection == S::INT8 => any::<i8>().prop_map(AnyValue::Int8).boxed(),
            _ if selection == S::INT16 => any::<i16>().prop_map(AnyValue::Int16).boxed(),
            _ if selection == S::INT32 => any::<i32>().prop_map(AnyValue::Int32).boxed(),
            _ if selection == S::INT64 => any::<i64>().prop_map(AnyValue::Int64).boxed(),
            _ if selection == S::INT128 => any::<i128>().prop_map(AnyValue::Int128).boxed(),
            _ if selection == S::FLOAT32 => any::<f32>().prop_map(AnyValue::Float32).boxed(),
            _ if selection == S::FLOAT64 => any::<f64>().prop_map(AnyValue::Float64).boxed(),
            #[cfg(feature = "dtype-date")]
            _ if selection == S::DATE => any::<i32>().prop_map(AnyValue::Date).boxed(),
            #[cfg(feature = "dtype-time")]
            _ if selection == S::TIME => any::<i64>().prop_map(AnyValue::Time).boxed(),
            _ if selection == S::BINARY => {
                prop::collection::vec(any::<u8>(), options.vector_length_range.clone())
                    .prop_map(|vec| {
                        let leaked: &'static [u8] = Box::leak(vec.into_boxed_slice());
                        AnyValue::Binary(leaked)
                    })
                    .boxed()
            },
            _ if selection == S::BINARY_OWNED => {
                prop::collection::vec(any::<u8>(), options.vector_length_range.clone())
                    .prop_map(AnyValue::BinaryOwned)
                    .boxed()
            },
            #[cfg(feature = "object")]
            _ if selection == S::OBJECT => object_strategy().boxed(),
            #[cfg(feature = "object")]
            _ if selection == S::OBJECT_OWNED => object_owned_strategy().boxed(),
            #[cfg(feature = "dtype-datetime")]
            _ if selection == S::DATETIME => datetime_strategy().boxed(),
            #[cfg(feature = "dtype-datetime")]
            _ if selection == S::DATETIME_OWNED => datetime_owned_strategy().boxed(),
            #[cfg(feature = "dtype-duration")]
            _ if selection == S::DURATION => duration_strategy().boxed(),
            #[cfg(feature = "dtype-decimal")]
            _ if selection == S::DECIMAL => {
                decimal_strategy(options.decimal_precision_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-categorical")]
            _ if selection == S::CATEGORICAL => {
                categorical_enum_strategy(options.categories_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-categorical")]
            _ if selection == S::CATEGORICAL_OWNED => {
                categorical_enum_owned_strategy(options.categories_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-categorical")]
            _ if selection == S::ENUM => {
                categorical_enum_strategy(options.categories_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-categorical")]
            _ if selection == S::ENUM_OWNED => {
                categorical_enum_owned_strategy(options.categories_range.clone()).boxed()
            },
            _ if selection == S::LIST => unimplemented!(),
            _ => unreachable!(), // TODO: Rest of strategies
        }
    })
}

impl PolarsObject for u64 {
    fn type_name() -> &'static str {
        "test_u64"
    }
}

impl PolarsObject for String {
    fn type_name() -> &'static str {
        "test_string"
    }
}

impl PolarsObject for bool {
    fn type_name() -> &'static str {
        "test_bool"
    }
}

#[cfg(feature = "object")]
fn object_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    proptest::prop_oneof![
        any::<u64>().prop_map(|u| {
            let boxed: Box<dyn PolarsObjectSafe> = Box::new(u);
            let leaked: &'static dyn PolarsObjectSafe = Box::leak(boxed);
            AnyValue::Object(leaked)
        }),
        any::<String>().prop_map(|s| {
            let boxed: Box<dyn PolarsObjectSafe> = Box::new(s);
            let leaked: &'static dyn PolarsObjectSafe = Box::leak(boxed);
            AnyValue::Object(leaked)
        }),
        any::<bool>().prop_map(|b| {
            let boxed: Box<dyn PolarsObjectSafe> = Box::new(b);
            let leaked: &'static dyn PolarsObjectSafe = Box::leak(boxed);
            AnyValue::Object(leaked)
        })
    ]
}

#[cfg(feature = "object")]
fn object_owned_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    proptest::prop_oneof![
        any::<u64>().prop_map(|u| {
            let boxed: Box<dyn PolarsObjectSafe> = Box::new(u);
            AnyValue::ObjectOwned(OwnedObject(boxed))
        }),
        any::<String>().prop_map(|s| {
            let boxed: Box<dyn PolarsObjectSafe> = Box::new(s);
            AnyValue::ObjectOwned(OwnedObject(boxed))
        }),
        any::<bool>().prop_map(|b| {
            let boxed: Box<dyn PolarsObjectSafe> = Box::new(b);
            AnyValue::ObjectOwned(OwnedObject(boxed))
        }),
    ]
}

#[cfg(feature = "dtype-datetime")]
fn datetime_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    (
        any::<i64>(),
        proptest::prop_oneof![
            Just(TimeUnit::Nanoseconds),
            Just(TimeUnit::Microseconds),
            Just(TimeUnit::Milliseconds),
        ],
    )
        .prop_map(|(timestamp, time_unit)| AnyValue::Datetime(timestamp, time_unit, None))
}

#[cfg(feature = "dtype-datetime")]
fn datetime_owned_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    (
        any::<i64>(),
        proptest::prop_oneof![
            Just(TimeUnit::Nanoseconds),
            Just(TimeUnit::Microseconds),
            Just(TimeUnit::Milliseconds),
        ],
    )
        .prop_map(|(timestamp, time_unit)| AnyValue::Datetime(timestamp, time_unit, None))
}

#[cfg(feature = "dtype-duration")]
fn duration_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    (
        any::<i64>(),
        proptest::prop_oneof![
            Just(TimeUnit::Nanoseconds),
            Just(TimeUnit::Microseconds),
            Just(TimeUnit::Milliseconds),
        ],
    )
        .prop_map(|(timestamp, time_unit)| AnyValue::Duration(timestamp, time_unit))
}

#[cfg(feature = "dtype-decimal")]
fn decimal_strategy(
    decimal_precision_range: RangeInclusive<usize>,
) -> impl Strategy<Value = AnyValue<'static>> {
    (any::<i128>(), decimal_precision_range)
        .prop_map(|(value, scale)| AnyValue::Decimal(value, scale))
}

#[cfg(feature = "dtype-categorical")]
fn categorical_enum_strategy(
    categories_range: RangeInclusive<usize>,
) -> impl Strategy<Value = AnyValue<'static>> {
    categories_range
        .prop_flat_map(|n_categories| (0..n_categories as u32, Just(n_categories)))
        .prop_map(|(cat_size, n_categories)| {
            let mapping = CategoricalMapping::new(n_categories.max(1));

            for i in 0..n_categories {
                let category_name = format!("category{i}");
                mapping.insert_cat(&category_name).unwrap();
            }

            let leaked_mapping: &'static Arc<CategoricalMapping> =
                Box::leak(Box::new(Arc::new(mapping)));

            AnyValue::Categorical(cat_size, leaked_mapping)
        })
}

#[cfg(feature = "dtype-categorical")]
fn categorical_enum_owned_strategy(
    categories_range: RangeInclusive<usize>,
) -> impl Strategy<Value = AnyValue<'static>> {
    categories_range
        .prop_flat_map(|n_categories| (0..n_categories as u32, Just(n_categories)))
        .prop_map(|(cat_size, n_categories)| {
            let mapping = CategoricalMapping::new(n_categories.max(1));

            for i in 0..n_categories {
                let category_name = format!("category{i}");
                mapping.insert_cat(&category_name).unwrap();
            }

            let arc_mapping = Arc::new(mapping);

            AnyValue::CategoricalOwned(cat_size, arc_mapping)
        })
}

fn _list_strategy() {
    // TODO
}

// TODO: More complex strategies
