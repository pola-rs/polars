use polars_dtype::categorical::{CategoricalPhysical, Categories, FrozenCategories};
use polars_utils::pl_str::PlSmallStr;
use proptest::prelude::*;
use rand::Rng;

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

const MAX_DECIMAL_PRECISION: usize = 38;
const MAX_CATEGORIES: usize = 3;

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
    prop::option::of(1_usize..=MAX_DECIMAL_PRECISION)
        .prop_flat_map(|precision| {
            let max_scale = precision.unwrap_or(MAX_DECIMAL_PRECISION);
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
    proptest::prop_oneof![
        Just(CategoricalPhysical::U8),
        Just(CategoricalPhysical::U16),
        Just(CategoricalPhysical::U32),
    ]
    .prop_map(|physical| {
        // **Note:** The Rust API documentation has a different definition of Categories::new(),
        // taking `name`, `physical`, and `gc`, whereas the codebase has the implementation that I used,
        // which takes `name`, `namespace`, and `physical`.
        let name = PlSmallStr::from_static("test_category");
        let namespace = PlSmallStr::from_static("test_namespace");

        let categories = Categories::new(name, namespace, physical);
        let mapping = categories.mapping();

        // **Note:*** Used `rand` for randomly selecting number of categories over
        // `proptest` since it was simpler and cleaner than having nested
        // `prop_map()`. Also, `rand` was already available in the `Cargo.toml`.
        let n_categories = rand::rng().random_range(1..=MAX_CATEGORIES);

        for i in 0..n_categories {
            let category_name = format!("category{i}");
            mapping.insert_cat(&category_name).unwrap();
        }

        DataType::Categorical(categories, mapping)
    })
}

// Sub-strategy for Enum DataType
fn enum_strategy() -> impl Strategy<Value = DataType> {
    (1usize..=MAX_CATEGORIES).prop_map(|n_categories| {
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

// TODO: Strategies for nested DataTypes (Array, List, Struct)
