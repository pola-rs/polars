use polars_dtype::categorical::{CategoricalPhysical, Categories, FrozenCategories};
use polars_utils::pl_str::PlSmallStr;
use proptest::prelude::*;

use super::{DataType, Field, TimeUnit, TimeZone};

// Simple DataTypes that don't take any parameters
const SIMPLE_DTYPES: &[DataType] = &[
    DataType::Boolean,
    DataType::UInt8,
    DataType::UInt16,
    DataType::UInt32,
    DataType::UInt64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::Int128,
    DataType::Float32,
    DataType::Float64,
    DataType::String,
    DataType::Binary,
    DataType::BinaryOffset,
    DataType::Date,
    DataType::Time,
    DataType::Null,
];

const DECIMAL_PRECISION_LIMIT: usize = 38;
const CATEGORIES_LIMIT: usize = 3;
const ARRAY_WIDTH_LIMIT: usize = 3;
const STRUCT_FIELDS_LIMIT: usize = 3;

// Strategy for generating non-nested Polars DataTypes
fn flat_dtype_strategy() -> impl Strategy<Value = DataType> {
    prop_oneof![
        prop::sample::select(SIMPLE_DTYPES.to_vec()),
        decimal_strategy(),
        datetime_strategy(),
        duration_strategy(),
        categorical_strategy(),
        enum_strategy(),
        object_strategy(),
    ]
}

// Sub-strategy for generating a Decimal Datatype
fn decimal_strategy() -> impl Strategy<Value = DataType> {
    prop::option::of(1_usize..=DECIMAL_PRECISION_LIMIT)
        .prop_flat_map(|precision| {
            let max_scale = precision.unwrap_or(DECIMAL_PRECISION_LIMIT);
            let scale_strategy = prop::option::of(0_usize..=max_scale);

            (Just(precision), scale_strategy)
        })
        .prop_map(|(precision, scale)| DataType::Decimal(precision, scale))
}

// Sub-strategy for generating a Datetime DataType
fn datetime_strategy() -> impl Strategy<Value = DataType> {
    proptest::prop_oneof![
        Just(TimeUnit::Nanoseconds),
        Just(TimeUnit::Microseconds),
        Just(TimeUnit::Milliseconds),
    ]
    .prop_map(|time_unit| DataType::Datetime(time_unit, None))
}

// Sub-strategy for generating a Duration DataType
fn duration_strategy() -> impl Strategy<Value = DataType> {
    proptest::prop_oneof![
        Just(TimeUnit::Nanoseconds),
        Just(TimeUnit::Microseconds),
        Just(TimeUnit::Milliseconds),
    ]
    .prop_map(|time_unit| DataType::Duration(time_unit))
}

// Sub-strategy for Categorical DataType
fn categorical_strategy() -> impl Strategy<Value = DataType> {
    (
        proptest::prop_oneof![
        Just(CategoricalPhysical::U8),
        Just(CategoricalPhysical::U16),
        Just(CategoricalPhysical::U32),
    ],
    1usize..=CATEGORIES_LIMIT
    )
    .prop_map(|(physical, n_categories)|{
        let name = PlSmallStr::from_static("test_category");
        let namespace = PlSmallStr::from_static("test_namespace");
        
        // **Note:** The Rust API documentation has a different definition of Categories::new(),
        // taking `name`, `physical`, and `gc`, whereas the codebase has the implementation that I used,
        // which takes `name`, `namespace`, and `physical`.
        let categories = Categories::new(name, namespace, physical);
        let mapping = categories.mapping();

        for i in 0..n_categories {
            let category_name = format!("category{i}");
            mapping.insert_cat(&category_name).unwrap();
        }

        DataType::Categorical(categories, mapping)
    })
}

// Sub-strategy for Enum DataType
fn enum_strategy() -> impl Strategy<Value = DataType> {
    (1usize..=CATEGORIES_LIMIT).prop_map(|n_categories| {
        let category_names: Vec<String> =
            (0..n_categories).map(|i| format!("category{i}")).collect();

        // **Note:** The Rust API documentation has a different definition of FrozenCategories::new(),
        // taking `physical` and `strings`, whereas the codebase has the implementation I used,
        // taking only `strings`.
        let frozen_categories =
            FrozenCategories::new(category_names.iter().map(|s| s.as_str())).unwrap();

        let mapping = frozen_categories.mapping().clone();
        DataType::Enum(frozen_categories, mapping)
    })
}

// Sub-strategy for Object DataType
// **Note:** In the Python code, an existing strategy for Object didn't exist
// and it was marked as a TODO until "various issues are solved", but I did
// implement it in the Rust version.
fn object_strategy() -> impl Strategy<Value = DataType> {
    Just(DataType::Object("test_object"))
}

// Sub-strategy for List DataType
fn list_strategy(inner: impl Strategy<Value = DataType>) -> impl Strategy<Value = DataType> {
    inner.prop_map(|inner | DataType::List(Box::new(inner)))
}

// Sub-strategy for Array DataType
fn array_strategy(inner: impl Strategy<Value = DataType>) -> impl Strategy<Value = DataType> {
    (inner, 1usize..=ARRAY_WIDTH_LIMIT).prop_map(|(inner, width)| {
        DataType::Array(Box::new(inner), width)
    })
}

// Sub-strategy for Struct DataType
fn struct_strategy(inner: impl Strategy<Value = DataType>) -> impl Strategy<Value = DataType> {
    prop::collection::vec(inner, 1usize..=STRUCT_FIELDS_LIMIT)
        .prop_map(|datatypes_vec| {
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

// TODO: Main strategy function with recursion
