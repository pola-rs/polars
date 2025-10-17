use std::ops::RangeInclusive;
use std::rc::Rc;
use std::sync::Arc;

use arrow::bitmap::bitmask::nth_set_bit_u32;
use polars_utils::pl_str::PlSmallStr;
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;

use crate::datatypes::PolarsObjectSafe;
#[cfg(feature = "dtype-categorical")]
use crate::prelude::CategoricalMapping;
use crate::prelude::{
    AnyValue, ArrowDataType, ArrowField, DataType, Field as PolarsField, OwnedObject, PolarsObject,
    StructArray, TimeUnit,
};
use crate::series::Series;

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct AnyValueArbitrarySelection: u32 {
        const NULL              = 1;
        const BOOLEAN           = 1 << 1;
        const STRING            = 1 << 2;
        const STRING_OWNED      = 1 << 3;
        const UINT              = 1 << 4;
        const INT               = 1 << 5;
        const FLOAT             = 1 << 6;
        const DATE              = 1 << 7;
        const TIME              = 1 << 8;
        const BINARY            = 1 << 9;
        const BINARY_OWNED      = 1 << 10;
        const OBJECT            = 1 << 11;
        const OBJECT_OWNED      = 1 << 12;

        const DATETIME          = 1 << 13;
        const DATETIME_OWNED    = 1 << 14;
        const DURATION          = 1 << 15;
        const DECIMAL           = 1 << 16;
        const CATEGORICAL       = 1 << 17;
        const CATEGORICAL_OWNED = 1 << 18;
        const ENUM              = 1 << 19;
        const ENUM_OWNED        = 1 << 20;

        const LIST              = 1 << 21;
        const ARRAY             = 1 << 22;
        const STRUCT            = 1 << 23;
        const STRUCT_OWNED      = 1 << 24;
    }
}

impl AnyValueArbitrarySelection {
    pub fn nested() -> Self {
        Self::LIST | Self::ARRAY | Self::STRUCT | Self::STRUCT_OWNED
    }
}

#[derive(Clone)]
pub struct AnyValueArbitraryOptions {
    pub allowed_dtypes: AnyValueArbitrarySelection,
    pub max_nesting_level: usize,
    pub vector_length_range: RangeInclusive<usize>,
    pub decimal_precision_range: RangeInclusive<usize>,
    pub categories_range: RangeInclusive<usize>,
    pub list_length: RangeInclusive<usize>,
    pub array_width_range: RangeInclusive<usize>,
    pub struct_fields_range: RangeInclusive<usize>,
}

impl Default for AnyValueArbitraryOptions {
    fn default() -> Self {
        Self {
            allowed_dtypes: AnyValueArbitrarySelection::all(),
            max_nesting_level: 3,
            vector_length_range: 0..=5,
            decimal_precision_range: 1..=38,
            categories_range: 0..=3,
            list_length: 0..=3,
            array_width_range: 0..=3,
            struct_fields_range: 0..=3,
        }
    }
}

pub fn anyvalue_strategy(
    options: Rc<AnyValueArbitraryOptions>,
    nesting_level: usize,
) -> impl Strategy<Value = AnyValue<'static>> {
    use AnyValueArbitrarySelection as S;
    let mut allowed_dtypes = options.allowed_dtypes;

    if options.max_nesting_level <= nesting_level {
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
            _ if selection == S::STRING => string_strategy().boxed(),
            _ if selection == S::STRING_OWNED => string_owned_strategy().boxed(),
            _ if selection == S::UINT => uint_strategy().boxed(),
            _ if selection == S::INT => int_strategy().boxed(),
            _ if selection == S::FLOAT => float_strategy().boxed(),
            #[cfg(feature = "dtype-date")]
            _ if selection == S::DATE => any::<i32>().prop_map(AnyValue::Date).boxed(),
            #[cfg(feature = "dtype-time")]
            _ if selection == S::TIME => any::<i64>().prop_map(AnyValue::Time).boxed(),
            _ if selection == S::BINARY => {
                binary_strategy(options.vector_length_range.clone()).boxed()
            },
            _ if selection == S::BINARY_OWNED => {
                binary_owned_strategy(options.vector_length_range.clone()).boxed()
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
            _ if selection == S::LIST => list_strategy(
                anyvalue_strategy(Rc::clone(&options), nesting_level + 1),
                options.list_length.clone(),
            )
            .boxed(),
            #[cfg(feature = "dtype-array")]
            _ if selection == S::ARRAY => array_strategy(
                anyvalue_strategy(Rc::clone(&options), nesting_level + 1),
                options.array_width_range.clone(),
            )
            .boxed(),
            #[cfg(feature = "dtype-struct")]
            _ if selection == S::STRUCT => struct_strategy(
                anyvalue_strategy(Rc::clone(&options), nesting_level + 1),
                options.struct_fields_range.clone(),
            ),
            #[cfg(feature = "dtype-struct")]
            _ if selection == S::STRUCT_OWNED => struct_owned_strategy(
                anyvalue_strategy(Rc::clone(&options), nesting_level + 1),
                options.struct_fields_range.clone(),
            ),
            _ => unreachable!(),
        }
    })
}

fn string_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    ".*".prop_map(|s| AnyValue::String(Box::leak(s.into_boxed_str())))
}

fn string_owned_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    any::<String>().prop_map(|s| AnyValue::StringOwned(PlSmallStr::from_string(s)))
}

fn uint_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    prop_oneof![
        any::<u8>().prop_map(AnyValue::UInt8),
        any::<u16>().prop_map(AnyValue::UInt16),
        any::<u32>().prop_map(AnyValue::UInt32),
        any::<u64>().prop_map(AnyValue::UInt64),
        any::<u128>().prop_map(AnyValue::UInt128),
    ]
}

fn int_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    prop_oneof![
        any::<i8>().prop_map(AnyValue::Int8),
        any::<i16>().prop_map(AnyValue::Int16),
        any::<i32>().prop_map(AnyValue::Int32),
        any::<i64>().prop_map(AnyValue::Int64),
        any::<i128>().prop_map(AnyValue::Int128),
    ]
}

fn float_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    prop_oneof![
        any::<f32>().prop_map(AnyValue::Float32),
        any::<f64>().prop_map(AnyValue::Float64),
    ]
}

fn binary_strategy(
    vector_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = AnyValue<'static>> {
    prop::collection::vec(any::<u8>(), vector_length_range).prop_map(|vec| {
        let leaked: &'static [u8] = Box::leak(vec.into_boxed_slice());
        AnyValue::Binary(leaked)
    })
}

fn binary_owned_strategy(
    vector_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = AnyValue<'static>> {
    prop::collection::vec(any::<u8>(), vector_length_range).prop_map(AnyValue::BinaryOwned)
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
        .prop_map(|(timestamp, time_unit)| AnyValue::DatetimeOwned(timestamp, time_unit, None))
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
    decimal_precision_range
        .prop_flat_map(|precision| (any::<i128>(), Just(precision), 0..=precision))
        .prop_map(|(value, precision, scale)| AnyValue::Decimal(value, precision, scale))
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

fn list_strategy(
    inner: impl Strategy<Value = AnyValue<'static>>,
    list_length: RangeInclusive<usize>,
) -> impl Strategy<Value = AnyValue<'static>> {
    inner.prop_flat_map(move |value| {
        let single_value = Just(value.clone());
        prop::collection::vec(single_value, list_length.clone()).prop_map(|anyvalue_values| {
            AnyValue::List(Series::from_any_values("".into(), &anyvalue_values, true).unwrap())
        })
    })
}

#[cfg(feature = "dtype-array")]
fn array_strategy(
    inner: impl Strategy<Value = AnyValue<'static>>,
    array_width_range: RangeInclusive<usize>,
) -> impl Strategy<Value = AnyValue<'static>> {
    inner.prop_flat_map(move |value| {
        let single_value = Just(value.clone());
        prop::collection::vec(single_value, array_width_range.clone()).prop_map(|anyvalue_values| {
            let series = Series::from_any_values("".into(), &anyvalue_values, true).unwrap();
            let len = series.len();
            AnyValue::Array(series, len)
        })
    })
}

#[cfg(feature = "dtype-struct")]
fn struct_strategy(
    inner: impl Strategy<Value = AnyValue<'static>> + 'static,
    struct_fields_range: RangeInclusive<usize>,
) -> BoxedStrategy<AnyValue<'static>> {
    prop::collection::vec(inner.boxed(), struct_fields_range)
        .prop_map(|field_values| {
            let fields: Vec<PolarsField> = field_values
                .iter()
                .enumerate()
                .map(|(i, _)| PolarsField::new(format!("field{i}").into(), DataType::Null))
                .collect();

            let arrow_fields: Vec<ArrowField> = fields
                .iter()
                .map(|f| ArrowField::new(f.name.clone(), ArrowDataType::Null, true))
                .collect();

            let struct_array = StructArray::new_empty(ArrowDataType::Struct(arrow_fields));
            let leaked_struct: &'static StructArray = Box::leak(Box::new(struct_array));
            let leaked_fields: &'static [PolarsField] = Box::leak(fields.into_boxed_slice());

            AnyValue::Struct(0, leaked_struct, leaked_fields)
        })
        .boxed()
}

#[cfg(feature = "dtype-struct")]
fn struct_owned_strategy(
    inner: impl Strategy<Value = AnyValue<'static>> + 'static,
    struct_fields_range: RangeInclusive<usize>,
) -> BoxedStrategy<AnyValue<'static>> {
    prop::collection::vec(inner.boxed(), struct_fields_range)
        .prop_map(|field_values| {
            let fields: Vec<PolarsField> = field_values
                .iter()
                .enumerate()
                .map(|(i, value)| PolarsField::new(format!("field{i}").into(), value.dtype()))
                .collect();

            AnyValue::StructOwned(Box::new((field_values, fields)))
        })
        .boxed()
}
