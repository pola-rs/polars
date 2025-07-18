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

// Create a strategy for generating non-nested Polars DataTypes
pub fn flat_dtypes() -> impl Strategy<Value = DataType> {
    prop_oneof![
        prop::sample::select(SIMPLE_DTYPES.to_vec()),
        decimal_strategy(),
        datetime_strategy(),
        duration_strategy(),
    ]
}

// Sub-strategy for generating a Decimal Datatype (since it takes additional inputs)
fn decimal_strategy() -> impl Strategy<Value = DataType> {
    prop::option::of(1_usize..=38)
        .prop_flat_map(|precision| {
            let max_scale = precision.unwrap_or(38);
            let scale_strategy = prop::option::of(0_usize..=max_scale);

            (Just(precision), scale_strategy)
        })
        .prop_map(|(precision, scale)| DataType::Decimal(precision, scale))
}

// Sub-strategy for generating a Datetime DataType (since it takes additional inputs)
fn datetime_strategy() -> impl Strategy<Value = DataType> {
    let time_unit_strategy = proptest::prop_oneof![
        Just(TimeUnit::Nanoseconds),
        Just(TimeUnit::Microseconds),
        Just(TimeUnit::Milliseconds),
    ];
    time_unit_strategy.prop_map(|time_unit| {
        DataType::Datetime(time_unit, None)
    })
}

// Sub-strategy for generating a Duration DataType (since it takes additional inputs)
fn duration_strategy() -> impl Strategy<Value = DataType> {
    let time_unit_strategy = proptest::prop_oneof![
        Just(TimeUnit::Nanoseconds),
        Just(TimeUnit::Microseconds),
        Just(TimeUnit::Milliseconds),
    ];
    time_unit_strategy.prop_map(|time_unit| {
        DataType::Duration(time_unit)
    })
}

// TODO: Sub-strategy for Categorical DataType which is complex:
// Categorical(Arc<Categories>, Arc<CategoricalMapping>)

// TODO: Sub-strategy for Enum DataType which is complex:
// Enum(Arc<FrozenCategories>, Arc<CategoricalMapping>)
