use std::ops::RangeInclusive;
use std::rc::Rc;

use arrow::bitmap::bitmask::nth_set_bit_u32;
#[cfg(feature = "dtype-categorical")]
use polars_dtype::categorical::{CategoricalPhysical, Categories, FrozenCategories};
use polars_utils::pl_str::PlSmallStr;
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;

use crate::datatypes::{PolarsObjectSafe, pf16};
use crate::prelude::{
    AnyValue, ArrowDataType, ArrowField, DataType, Field, OwnedObject, PolarsObject, StructArray,
    TimeUnit,
};
use crate::series::Series;

// DataType Strategies

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DataTypeArbitrarySelection: u32 {
        const BOOLEAN = 1;
        const UINT = 1 << 1;
        const INT = 1 << 2;
        const FLOAT = 1 << 3;
        const STRING = 1 << 4;
        const BINARY = 1 << 5;
        const BINARY_OFFSET = 1 << 6;
        const DATE = 1 << 7;
        const TIME = 1 << 8;
        const NULL = 1 << 9;

        const DECIMAL = 1 << 10;
        const DATETIME = 1 << 11;
        const DURATION = 1 << 12;
        const CATEGORICAL = 1 << 13;
        const ENUM = 1 << 14;
        const OBJECT = 1 << 15;

        const LIST = 1 << 16;
        const ARRAY = 1 << 17;
        const STRUCT = 1 << 18;
    }
}

impl DataTypeArbitrarySelection {
    pub fn nested() -> Self {
        Self::LIST | Self::ARRAY | Self::STRUCT
    }
}

#[derive(Clone)]
pub struct DataTypeArbitraryOptions {
    pub allowed_dtypes: DataTypeArbitrarySelection,
    pub max_nesting_level: usize,
    pub decimal_precision_range: RangeInclusive<usize>,
    pub categories_range: RangeInclusive<usize>,
    pub array_width_range: RangeInclusive<usize>,
    pub struct_fields_range: RangeInclusive<usize>,
}

impl Default for DataTypeArbitraryOptions {
    fn default() -> Self {
        Self {
            allowed_dtypes: DataTypeArbitrarySelection::all(),
            max_nesting_level: 3,
            decimal_precision_range: 1..=38,
            categories_range: 0..=3,
            array_width_range: 0..=3,
            struct_fields_range: 0..=3,
        }
    }
}

pub fn dtypes_strategy(
    options: Rc<DataTypeArbitraryOptions>,
    nesting_level: usize,
) -> impl Strategy<Value = DataType> {
    use DataTypeArbitrarySelection as S;
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
            _ if selection == S::BOOLEAN => Just(DataType::Boolean).boxed(),
            _ if selection == S::UINT => dtype_uint_strategy().boxed(),
            _ if selection == S::INT => dtype_int_strategy().boxed(),
            _ if selection == S::FLOAT => dtype_float_strategy().boxed(),
            _ if selection == S::STRING => Just(DataType::String).boxed(),
            _ if selection == S::BINARY => Just(DataType::Binary).boxed(),
            _ if selection == S::BINARY_OFFSET => Just(DataType::BinaryOffset).boxed(),
            _ if selection == S::DATE => Just(DataType::Date).boxed(),
            _ if selection == S::TIME => Just(DataType::Time).boxed(),
            _ if selection == S::NULL => Just(DataType::Null).boxed(),
            #[cfg(feature = "dtype-decimal")]
            _ if selection == S::DECIMAL => {
                dtype_decimal_strategy(options.decimal_precision_range.clone()).boxed()
            },
            _ if selection == S::DATETIME => dtype_datetime_strategy().boxed(),
            _ if selection == S::DURATION => dtype_duration_strategy().boxed(),
            #[cfg(feature = "dtype-categorical")]
            _ if selection == S::CATEGORICAL => {
                dtype_categorical_strategy(options.categories_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-categorical")]
            _ if selection == S::ENUM => {
                dtype_enum_strategy(options.categories_range.clone()).boxed()
            },
            #[cfg(feature = "object")]
            _ if selection == S::OBJECT => Just(DataType::Object("test_object")).boxed(),
            _ if selection == S::LIST => {
                dtype_list_strategy(dtypes_strategy(Rc::clone(&options), nesting_level + 1)).boxed()
            },
            #[cfg(feature = "dtype-array")]
            _ if selection == S::ARRAY => dtype_array_strategy(
                dtypes_strategy(Rc::clone(&options), nesting_level + 1),
                options.array_width_range.clone(),
            )
            .boxed(),
            #[cfg(feature = "dtype-struct")]
            _ if selection == S::STRUCT => dtype_struct_strategy(
                dtypes_strategy(Rc::clone(&options), nesting_level + 1),
                options.struct_fields_range.clone(),
            )
            .boxed(),
            _ => unreachable!(),
        }
    })
}

fn dtype_uint_strategy() -> impl Strategy<Value = DataType> {
    prop_oneof![
        Just(DataType::UInt8),
        Just(DataType::UInt16),
        Just(DataType::UInt32),
        Just(DataType::UInt64),
        Just(DataType::UInt128),
    ]
}

fn dtype_int_strategy() -> impl Strategy<Value = DataType> {
    prop_oneof![
        Just(DataType::Int8),
        Just(DataType::Int16),
        Just(DataType::Int32),
        Just(DataType::Int64),
        Just(DataType::Int128),
    ]
}

fn dtype_float_strategy() -> impl Strategy<Value = DataType> {
    prop_oneof![
        Just(DataType::Float16),
        Just(DataType::Float32),
        Just(DataType::Float64),
    ]
}

#[cfg(feature = "dtype-decimal")]
fn dtype_decimal_strategy(
    decimal_precision_range: RangeInclusive<usize>,
) -> impl Strategy<Value = DataType> {
    decimal_precision_range
        .clone()
        .prop_flat_map(move |precision| {
            let scale_strategy = 0_usize..=precision;
            (Just(precision), scale_strategy)
        })
        .prop_map(|(precision, scale)| DataType::Decimal(precision, scale))
}

fn dtype_datetime_strategy() -> impl Strategy<Value = DataType> {
    proptest::prop_oneof![
        Just(TimeUnit::Nanoseconds),
        Just(TimeUnit::Microseconds),
        Just(TimeUnit::Milliseconds),
    ]
    .prop_map(|time_unit| DataType::Datetime(time_unit, None))
}

fn dtype_duration_strategy() -> impl Strategy<Value = DataType> {
    proptest::prop_oneof![
        Just(TimeUnit::Nanoseconds),
        Just(TimeUnit::Microseconds),
        Just(TimeUnit::Milliseconds),
    ]
    .prop_map(DataType::Duration)
}

#[cfg(feature = "dtype-categorical")]
fn dtype_categorical_strategy(
    categories_range: RangeInclusive<usize>,
) -> impl Strategy<Value = DataType> {
    (
        proptest::prop_oneof![
            Just(CategoricalPhysical::U8),
            Just(CategoricalPhysical::U16),
            Just(CategoricalPhysical::U32),
        ],
        categories_range,
    )
        .prop_map(|(physical, n_categories)| {
            let name = PlSmallStr::from_static("test_category");
            let namespace = PlSmallStr::from_static("test_namespace");

            let categories = Categories::new(name, namespace, physical);
            let mapping = categories.mapping();

            for i in 0..n_categories {
                let category_name = format!("category{i}");
                mapping.insert_cat(&category_name).unwrap();
            }

            DataType::Categorical(categories, mapping)
        })
}

#[cfg(feature = "dtype-categorical")]
fn dtype_enum_strategy(categories_range: RangeInclusive<usize>) -> impl Strategy<Value = DataType> {
    (categories_range).prop_map(|n_categories| {
        let category_names: Vec<String> =
            (0..n_categories).map(|i| format!("category{i}")).collect();

        let frozen_categories =
            FrozenCategories::new(category_names.iter().map(|s| s.as_str())).unwrap();

        let mapping = frozen_categories.mapping().clone();
        DataType::Enum(frozen_categories, mapping)
    })
}

fn dtype_list_strategy(inner: impl Strategy<Value = DataType>) -> impl Strategy<Value = DataType> {
    inner.prop_map(|inner| DataType::List(Box::new(inner)))
}

#[cfg(feature = "dtype-array")]
fn dtype_array_strategy(
    inner: impl Strategy<Value = DataType>,
    array_width_range: RangeInclusive<usize>,
) -> impl Strategy<Value = DataType> {
    (inner, array_width_range).prop_map(|(inner, width)| DataType::Array(Box::new(inner), width))
}

#[cfg(feature = "dtype-struct")]
fn dtype_struct_strategy(
    inner: impl Strategy<Value = DataType>,
    struct_fields_range: RangeInclusive<usize>,
) -> impl Strategy<Value = DataType> {
    prop::collection::vec(inner, struct_fields_range).prop_map(|fields_values| {
        let fields: Vec<Field> = fields_values
            .into_iter()
            .enumerate()
            .map(|(i, datatype)| Field::new(format!("field{i}").into(), datatype))
            .collect();

        DataType::Struct(fields)
    })
}

// AnyValue Strategies

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
            _ if selection == S::STRING => av_string_strategy().boxed(),
            _ if selection == S::STRING_OWNED => av_string_owned_strategy().boxed(),
            _ if selection == S::UINT => av_uint_strategy().boxed(),
            _ if selection == S::INT => av_int_strategy().boxed(),
            _ if selection == S::FLOAT => av_float_strategy().boxed(),
            #[cfg(feature = "dtype-date")]
            _ if selection == S::DATE => any::<i32>().prop_map(AnyValue::Date).boxed(),
            #[cfg(feature = "dtype-time")]
            _ if selection == S::TIME => any::<i64>().prop_map(AnyValue::Time).boxed(),
            _ if selection == S::BINARY => {
                av_binary_strategy(options.vector_length_range.clone()).boxed()
            },
            _ if selection == S::BINARY_OWNED => {
                av_binary_owned_strategy(options.vector_length_range.clone()).boxed()
            },
            #[cfg(feature = "object")]
            _ if selection == S::OBJECT => av_object_strategy().boxed(),
            #[cfg(feature = "object")]
            _ if selection == S::OBJECT_OWNED => av_object_owned_strategy().boxed(),
            #[cfg(feature = "dtype-datetime")]
            _ if selection == S::DATETIME => av_datetime_strategy().boxed(),
            #[cfg(feature = "dtype-datetime")]
            _ if selection == S::DATETIME_OWNED => av_datetime_owned_strategy().boxed(),
            #[cfg(feature = "dtype-duration")]
            _ if selection == S::DURATION => av_duration_strategy().boxed(),
            #[cfg(feature = "dtype-decimal")]
            _ if selection == S::DECIMAL => {
                av_decimal_strategy(options.decimal_precision_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-categorical")]
            _ if selection == S::CATEGORICAL => {
                av_categorical_enum_strategy(options.categories_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-categorical")]
            _ if selection == S::CATEGORICAL_OWNED => {
                av_categorical_enum_owned_strategy(options.categories_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-categorical")]
            _ if selection == S::ENUM => {
                av_categorical_enum_strategy(options.categories_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-categorical")]
            _ if selection == S::ENUM_OWNED => {
                av_categorical_enum_owned_strategy(options.categories_range.clone()).boxed()
            },
            _ if selection == S::LIST => av_list_strategy(
                anyvalue_strategy(Rc::clone(&options), nesting_level + 1),
                options.list_length.clone(),
            )
            .boxed(),
            #[cfg(feature = "dtype-array")]
            _ if selection == S::ARRAY => av_array_strategy(
                anyvalue_strategy(Rc::clone(&options), nesting_level + 1),
                options.array_width_range.clone(),
            )
            .boxed(),
            #[cfg(feature = "dtype-struct")]
            _ if selection == S::STRUCT => av_struct_strategy(
                anyvalue_strategy(Rc::clone(&options), nesting_level + 1),
                options.struct_fields_range.clone(),
            ),
            #[cfg(feature = "dtype-struct")]
            _ if selection == S::STRUCT_OWNED => av_struct_owned_strategy(
                anyvalue_strategy(Rc::clone(&options), nesting_level + 1),
                options.struct_fields_range.clone(),
            ),
            _ => unreachable!(),
        }
    })
}

fn av_string_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    ".*".prop_map(|s| AnyValue::String(Box::leak(s.into_boxed_str())))
}

fn av_string_owned_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    any::<String>().prop_map(|s| AnyValue::StringOwned(PlSmallStr::from_string(s)))
}

fn av_uint_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    prop_oneof![
        any::<u8>().prop_map(AnyValue::UInt8),
        any::<u16>().prop_map(AnyValue::UInt16),
        any::<u32>().prop_map(AnyValue::UInt32),
        any::<u64>().prop_map(AnyValue::UInt64),
        any::<u128>().prop_map(AnyValue::UInt128),
    ]
}

fn av_int_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    prop_oneof![
        any::<i8>().prop_map(AnyValue::Int8),
        any::<i16>().prop_map(AnyValue::Int16),
        any::<i32>().prop_map(AnyValue::Int32),
        any::<i64>().prop_map(AnyValue::Int64),
        any::<i128>().prop_map(AnyValue::Int128),
    ]
}

fn av_float_strategy() -> impl Strategy<Value = AnyValue<'static>> {
    prop_oneof![
        any::<u16>().prop_map(|bits| AnyValue::Float16(pf16::from_bits(bits))),
        any::<f32>().prop_map(AnyValue::Float32),
        any::<f64>().prop_map(AnyValue::Float64),
    ]
}

fn av_binary_strategy(
    vector_length_range: RangeInclusive<usize>,
) -> impl Strategy<Value = AnyValue<'static>> {
    prop::collection::vec(any::<u8>(), vector_length_range).prop_map(|vec| {
        let leaked: &'static [u8] = Box::leak(vec.into_boxed_slice());
        AnyValue::Binary(leaked)
    })
}

fn av_binary_owned_strategy(
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
fn av_object_strategy() -> impl Strategy<Value = AnyValue<'static>> {
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
fn av_object_owned_strategy() -> impl Strategy<Value = AnyValue<'static>> {
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
fn av_datetime_strategy() -> impl Strategy<Value = AnyValue<'static>> {
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
fn av_datetime_owned_strategy() -> impl Strategy<Value = AnyValue<'static>> {
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
fn av_duration_strategy() -> impl Strategy<Value = AnyValue<'static>> {
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
fn av_decimal_strategy(
    decimal_precision_range: RangeInclusive<usize>,
) -> impl Strategy<Value = AnyValue<'static>> {
    decimal_precision_range
        .prop_flat_map(|precision| {
            let max_value = 10_i128.pow(precision as u32) - 1;
            let min_value = -max_value;

            (min_value..=max_value, Just(precision), 0..=precision)
        })
        .prop_map(|(value, precision, scale)| AnyValue::Decimal(value, precision, scale))
}

#[cfg(feature = "dtype-categorical")]
fn av_categorical_enum_strategy(
    categories_range: RangeInclusive<usize>,
) -> impl Strategy<Value = AnyValue<'static>> {
    categories_range
        .prop_flat_map(|n_categories| (0..n_categories as u32, Just(n_categories)))
        .prop_map(|(cat_size, n_categories)| {
            let cat = Categories::random(PlSmallStr::EMPTY, CategoricalPhysical::U32);
            let mapping = cat.mapping();

            for i in 0..n_categories {
                let category_name = format!("category{i}");
                mapping.insert_cat(&category_name).unwrap();
            }

            let leaked_mapping = Box::leak(Box::new(mapping));
            AnyValue::Categorical(cat_size, leaked_mapping)
        })
}

#[cfg(feature = "dtype-categorical")]
fn av_categorical_enum_owned_strategy(
    categories_range: RangeInclusive<usize>,
) -> impl Strategy<Value = AnyValue<'static>> {
    categories_range
        .prop_flat_map(|n_categories| (0..n_categories as u32, Just(n_categories)))
        .prop_map(|(cat_size, n_categories)| {
            let cat = Categories::random(PlSmallStr::EMPTY, CategoricalPhysical::U32);
            let mapping = cat.mapping();

            for i in 0..n_categories {
                let category_name = format!("category{i}");
                mapping.insert_cat(&category_name).unwrap();
            }

            AnyValue::CategoricalOwned(cat_size, mapping)
        })
}

fn av_list_strategy(
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
fn av_array_strategy(
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
fn av_struct_strategy(
    inner: impl Strategy<Value = AnyValue<'static>> + 'static,
    struct_fields_range: RangeInclusive<usize>,
) -> BoxedStrategy<AnyValue<'static>> {
    prop::collection::vec(inner.boxed(), struct_fields_range)
        .prop_map(|field_values| {
            let fields: Vec<Field> = field_values
                .iter()
                .enumerate()
                .map(|(i, _)| Field::new(format!("field{i}").into(), DataType::Null))
                .collect();

            let arrow_fields: Vec<ArrowField> = fields
                .iter()
                .map(|f| ArrowField::new(f.name.clone(), ArrowDataType::Null, true))
                .collect();

            let struct_array = StructArray::new_empty(ArrowDataType::Struct(arrow_fields));
            let leaked_struct: &'static StructArray = Box::leak(Box::new(struct_array));
            let leaked_fields: &'static [Field] = Box::leak(fields.into_boxed_slice());

            AnyValue::Struct(0, leaked_struct, leaked_fields)
        })
        .boxed()
}

#[cfg(feature = "dtype-struct")]
fn av_struct_owned_strategy(
    inner: impl Strategy<Value = AnyValue<'static>> + 'static,
    struct_fields_range: RangeInclusive<usize>,
) -> BoxedStrategy<AnyValue<'static>> {
    prop::collection::vec(inner.boxed(), struct_fields_range)
        .prop_map(|field_values| {
            let fields: Vec<Field> = field_values
                .iter()
                .enumerate()
                .map(|(i, value)| Field::new(format!("field{i}").into(), value.dtype()))
                .collect();

            AnyValue::StructOwned(Box::new((field_values, fields)))
        })
        .boxed()
}
