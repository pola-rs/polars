use std::ops::RangeInclusive;
use std::rc::Rc;

use arrow::bitmap::bitmask::nth_set_bit_u32;
#[cfg(feature = "dtype-categorical")]
use polars_dtype::categorical::{CategoricalPhysical, Categories, FrozenCategories};
use polars_utils::pl_str::PlSmallStr;
use proptest::prelude::*;

use super::{DataType, Field, TimeUnit};

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DataTypeArbitrarySelection: u32 {
        const BOOLEAN = 1;
        const UINT8 = 1 << 1;
        const UINT16 = 1 << 2;
        const UINT32 = 1 << 3;
        const UINT64 = 1 << 4;
        const INT8 = 1 << 5;
        const INT16 = 1 << 6;
        const INT32 = 1 << 7;
        const INT64 = 1 << 8;
        const INT128 = 1 << 9;
        const FLOAT32 = 1 << 10;
        const FLOAT64 = 1 << 11;
        const STRING = 1 << 12;
        const BINARY = 1 << 13;
        const BINARY_OFFSET = 1 << 14;
        const DATE = 1 << 15;
        const TIME = 1 << 16;
        const NULL = 1 << 17;

        const DECIMAL = 1 << 18;
        const DATETIME = 1 << 19;
        const DURATION = 1 << 20;
        const CATEGORICAL = 1 << 21;
        const ENUM = 1 << 22;
        const OBJECT = 1 << 23;

        const LIST = 1 << 24;
        const ARRAY = 1 << 25;
        const STRUCT = 1 << 26;
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

pub fn dtypes(
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
            S::from_bits_retain(1 << nth_set_bit_u32(options.allowed_dtypes.bits(), i).unwrap());

        match selection {
            _ if selection == S::BOOLEAN => Just(DataType::Boolean).boxed(),
            _ if selection == S::UINT8 => Just(DataType::UInt8).boxed(),
            _ if selection == S::UINT16 => Just(DataType::UInt16).boxed(),
            _ if selection == S::UINT32 => Just(DataType::UInt32).boxed(),
            _ if selection == S::UINT64 => Just(DataType::UInt64).boxed(),
            _ if selection == S::INT8 => Just(DataType::Int8).boxed(),
            _ if selection == S::INT16 => Just(DataType::Int16).boxed(),
            _ if selection == S::INT32 => Just(DataType::Int32).boxed(),
            _ if selection == S::INT64 => Just(DataType::Int64).boxed(),
            _ if selection == S::INT128 => Just(DataType::Int128).boxed(),
            _ if selection == S::FLOAT32 => Just(DataType::Float32).boxed(),
            _ if selection == S::FLOAT64 => Just(DataType::Float64).boxed(),
            _ if selection == S::STRING => Just(DataType::String).boxed(),
            _ if selection == S::BINARY => Just(DataType::Binary).boxed(),
            _ if selection == S::BINARY_OFFSET => Just(DataType::BinaryOffset).boxed(),
            _ if selection == S::DATE => Just(DataType::Date).boxed(),
            _ if selection == S::TIME => Just(DataType::Time).boxed(),
            _ if selection == S::NULL => Just(DataType::Null).boxed(),
            #[cfg(feature = "dtype-decimal")]
            _ if selection == S::DECIMAL => {
                decimal_strategy(options.decimal_precision_range.clone()).boxed()
            },
            _ if selection == S::DATETIME => datetime_strategy().boxed(),
            _ if selection == S::DURATION => duration_strategy().boxed(),
            #[cfg(feature = "dtype-categorical")]
            _ if selection == S::CATEGORICAL => {
                categorical_strategy(options.categories_range.clone()).boxed()
            },
            #[cfg(feature = "dtype-categorical")]
            _ if selection == S::ENUM => enum_strategy(options.categories_range.clone()).boxed(),
            #[cfg(feature = "object")]
            _ if selection == S::OBJECT => Just(DataType::Object("test_object")).boxed(),
            _ if selection == S::LIST => {
                list_strategy(dtypes(options.clone(), nesting_level + 1)).boxed()
            },
            #[cfg(feature = "dtype-array")]
            _ if selection == S::ARRAY => array_strategy(
                dtypes(options.clone(), nesting_level + 1),
                options.array_width_range.clone(),
            )
            .boxed(),
            #[cfg(feature = "dtype-struct")]
            _ if selection == S::STRUCT => struct_strategy(
                dtypes(options.clone(), nesting_level + 1),
                options.struct_fields_range.clone(),
            )
            .boxed(),
            _ => unreachable!(),
        }
    })
}

#[cfg(feature = "dtype-decimal")]
fn decimal_strategy(
    decimal_precision_range: RangeInclusive<usize>,
) -> impl Strategy<Value = DataType> {
    prop::option::of(decimal_precision_range.clone())
        .prop_flat_map(move |precision| {
            let max_scale = precision.unwrap_or(*decimal_precision_range.end());
            let scale_strategy = prop::option::of(0_usize..=max_scale);

            (Just(precision), scale_strategy)
        })
        .prop_map(|(precision, scale)| DataType::Decimal(precision, scale))
}

fn datetime_strategy() -> impl Strategy<Value = DataType> {
    proptest::prop_oneof![
        Just(TimeUnit::Nanoseconds),
        Just(TimeUnit::Microseconds),
        Just(TimeUnit::Milliseconds),
    ]
    .prop_map(|time_unit| DataType::Datetime(time_unit, None))
}

fn duration_strategy() -> impl Strategy<Value = DataType> {
    proptest::prop_oneof![
        Just(TimeUnit::Nanoseconds),
        Just(TimeUnit::Microseconds),
        Just(TimeUnit::Milliseconds),
    ]
    .prop_map(DataType::Duration)
}

#[cfg(feature = "dtype-categorical")]
fn categorical_strategy(
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
fn enum_strategy(categories_range: RangeInclusive<usize>) -> impl Strategy<Value = DataType> {
    (categories_range).prop_map(|n_categories| {
        let category_names: Vec<String> =
            (0..n_categories).map(|i| format!("category{i}")).collect();

        let frozen_categories =
            FrozenCategories::new(category_names.iter().map(|s| s.as_str())).unwrap();

        let mapping = frozen_categories.mapping().clone();
        DataType::Enum(frozen_categories, mapping)
    })
}

fn list_strategy(inner: impl Strategy<Value = DataType>) -> impl Strategy<Value = DataType> {
    inner.prop_map(|inner| DataType::List(Box::new(inner)))
}

#[cfg(feature = "dtype-array")]
fn array_strategy(
    inner: impl Strategy<Value = DataType>,
    array_width_range: RangeInclusive<usize>,
) -> impl Strategy<Value = DataType> {
    (inner, array_width_range).prop_map(|(inner, width)| DataType::Array(Box::new(inner), width))
}

#[cfg(feature = "dtype-struct")]
fn struct_strategy(
    inner: impl Strategy<Value = DataType>,
    struct_fields_range: RangeInclusive<usize>,
) -> impl Strategy<Value = DataType> {
    prop::collection::vec(inner, struct_fields_range).prop_map(|datatypes_vec| {
        let fields_vec: Vec<Field> = datatypes_vec
            .into_iter()
            .enumerate()
            .map(|(i, datatype)| {
                let field_name = format!("field{i}");
                Field::new(field_name.into(), datatype)
            })
            .collect();

        DataType::Struct(fields_vec)
    })
}
