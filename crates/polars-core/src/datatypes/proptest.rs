use polars_dtype::categorical::{CategoricalPhysical, Categories, FrozenCategories};
use polars_utils::pl_str::PlSmallStr;
use proptest::prelude::*;

use super::{DataType, Field, TimeUnit};

// This code creates a struct that acts like a smart integer for representing sets of data types
// It is essentially a fancy 32-bit integer where each bit represents whether a specific data type is selected or not
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DataTypeArbitrarySelection: u32 {
        // Simple DataTypes that don't take any parameters
        const BOOLEAN = 1;              // bit 0
        const UINT8 = 1 << 1;          // bit 1
        const UINT16 = 1 << 2;         // bit 2 
        const UINT32 = 1 << 3;         // bit 3
        const UINT64 = 1 << 4;         // bit 4
        const INT8 = 1 << 5;           // bit 5
        const INT16 = 1 << 6;          // bit 6
        const INT32 = 1 << 7;          // bit 7
        const INT64 = 1 << 8;          // bit 8
        const INT128 = 1 << 9;         // bit 9
        const FLOAT32 = 1 << 10;       // bit 10
        const FLOAT64 = 1 << 11;       // bit 11
        const STRING = 1 << 12;        // bit 12
        const BINARY = 1 << 13;        // bit 13
        const BINARY_OFFSET = 1 << 14; // bit 14
        const DATE = 1 << 15;          // bit 15
        const TIME = 1 << 16;          // bit 16
        const NULL = 1 << 17;          // bit 17

        // Complex DataTypes that do take parameters
        const DECIMAL = 1 << 18;       // bit 18
        const DATETIME = 1 << 19;      // bit 19
        const DURATION = 1 << 20;      // bit 20
        const CATEGORICAL = 1 << 21;   // bit 21
        const ENUM = 1 << 22;          // bit 22
        const OBJECT = 1 << 23;        // bit 23

        // Nested DataTypes
        const LIST = 1 << 24;          // bit 24
        const ARRAY = 1 << 25;         // bit 25
        const STRUCT = 1 << 26;        // bit 26
    }
}

// Adding convenience methods to select different groups of DataTypes
// This is a static method (associated function) - it doesn't need an existing instance to work
// it's creating and returning a new instance, as opposed to instance methods
// Static methods are functions that belong to the type but don't need an existing instance to work
impl DataTypeArbitrarySelection {
    pub fn simple() -> Self {
        Self::BOOLEAN | Self::UINT8 | Self::UINT16 | Self::UINT32 | 
        Self::UINT64 | Self::INT8 | Self::INT16 | Self::INT32 | 
        Self::INT64 | Self::INT128 | Self::FLOAT32 | Self::FLOAT64 | 
        Self::STRING | Self::BINARY | Self::BINARY_OFFSET | 
        Self::DATE | Self::TIME | Self::NULL
    }

    pub fn complex() -> Self {
        Self::DECIMAL | Self::DATETIME | Self::DURATION | 
        Self::CATEGORICAL | Self::ENUM | Self::OBJECT
    }

    pub fn nested() -> Self {
        Self::LIST | Self::ARRAY | Self::STRUCT
    }
}

#[derive(Clone)]
pub struct DataTypeArbitraryOptions {
    pub allowed_dtypes: DataTypeArbitrarySelection,
    pub decimal_precision_limit: usize,
    pub categories_limit: usize,
    pub array_width_limit: usize,
    pub struct_fields_limit: usize,
    pub max_nesting_level: usize,
}

impl Default for DataTypeArbitraryOptions {
    fn default() -> Self {
        Self {
            allowed_dtypes: DataTypeArbitrarySelection::all(),
            decimal_precision_limit: 38,
            categories_limit: 3,
            array_width_limit: 3,
            struct_fields_limit: 3,
            max_nesting_level: 3,
        }
    }
}

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

// Main strategy function with recursion
// https://docs.rs/proptest/latest/proptest/strategy/trait.Strategy.html#method.prop_recursive
pub fn dtypes(nesting_level: u32) -> impl Strategy<Value = DataType> {
    // `flat_dtype_strategy()` is the base/leaf strategy
    flat_dtype_strategy().prop_recursive(
        nesting_level, // Maximum recursive depth; 0 means just the base strategy/leaves
        // Adding more depth means increasing the pool of possiblities
        // depth 1 would include 1 level of nesting as a possibility (List(Int64))
        // depth 2 would include 2 levels of nesting as a possibility (List(List(Int64)))
        // etc.
        // Seems that depth = maximum possibility, not a guarantee of nesting
        // Even with a high depth, selecting a base case with no nesting is still possible
        // From documentation: "Generated structures will have a depth between 0 and depth..."
        256, // desired_size: Target total number of elements in the generated structure
        // (influences how big arrays/lists/structs become to reach this target)
        10, // expected_branch_size: Expected number of items per collection/container
        // (controls how many fields in structs, items in arrays/lists, etc.)
        // `nested_dtype_strategy()` is the recursive strategy
        nested_dtype_strategy,
    )
}
// **Note:** Can make `desired_size` and `expected_branch_size` configurable
// by the end-user or set it to sensible defaults.

// Sub-strategy for generating non-nested Polars DataTypes
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
    .prop_map(DataType::Duration)
}

// Sub-strategy for Categorical DataType
fn categorical_strategy() -> impl Strategy<Value = DataType> {
    (
        proptest::prop_oneof![
            Just(CategoricalPhysical::U8),
            Just(CategoricalPhysical::U16),
            Just(CategoricalPhysical::U32),
        ],
        1usize..=CATEGORIES_LIMIT,
    )
        .prop_map(|(physical, n_categories)| {
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

// Sub-strategy for generating nested Polars DataTypes
fn nested_dtype_strategy(
    inner: impl Strategy<Value = DataType> + Clone,
) -> impl Strategy<Value = DataType> {
    proptest::prop_oneof![
        list_strategy(inner.clone()),
        array_strategy(inner.clone()),
        struct_strategy(inner.clone()),
    ]
}

// Sub-strategy for List DataType
fn list_strategy(inner: impl Strategy<Value = DataType>) -> impl Strategy<Value = DataType> {
    inner.prop_map(|inner| DataType::List(Box::new(inner)))
}

// Sub-strategy for Array DataType
fn array_strategy(inner: impl Strategy<Value = DataType>) -> impl Strategy<Value = DataType> {
    (inner, 1usize..=ARRAY_WIDTH_LIMIT)
        .prop_map(|(inner, width)| DataType::Array(Box::new(inner), width))
}

// Sub-strategy for Struct DataType
fn struct_strategy(inner: impl Strategy<Value = DataType>) -> impl Strategy<Value = DataType> {
    prop::collection::vec(inner, 1usize..=STRUCT_FIELDS_LIMIT).prop_map(|datatypes_vec| {
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
