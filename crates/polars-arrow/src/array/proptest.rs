use std::ops::RangeInclusive;
use std::rc::Rc;

use polars_utils::format_pl_smallstr;
use proptest::prelude::{Just, Strategy};
use proptest::sample::SizeRange;

use super::binview::proptest::binview_array;
use super::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeListArray, ListArray, NullArray,
    StructArray,
};
use crate::array::binview::proptest::utf8view_array;
use crate::array::boolean::proptest::boolean_array;
use crate::array::primitive::proptest::primitive_array;
use crate::array::{PrimitiveArray, Utf8ViewArray};
use crate::bitmap::bitmask::nth_set_bit_u32;
use crate::datatypes::{ArrowDataType, Field};

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ArrowDataTypeArbitrarySelection: u32 {
        const NULL = 1;

        const BOOLEAN = 1 << 1;

        const INT8 = 1 << 2;
        const INT16 = 1 << 3;
        const INT32 = 1 << 4;
        const INT64 = 1 << 5;
        const INT128 = 1 << 6;

        const UINT8 = 1 << 7;
        const UINT16 = 1 << 8;
        const UINT32 = 1 << 9;
        const UINT64 = 1 << 10;

        const FLOAT32 = 1 << 11;
        const FLOAT64 = 1 << 12;

        const STRVIEW = 1 << 13;
        const BINVIEW = 1 << 14;
        const BINARY = 1 << 15;

        const LIST = 1 << 16;
        const FIXED_SIZE_LIST = 1 << 17;
        const STRUCT = 1 << 18;
    }
}

impl ArrowDataTypeArbitrarySelection {
    pub fn nested() -> Self {
        Self::LIST | Self::FIXED_SIZE_LIST | Self::STRUCT
    }
}

#[derive(Clone)]
pub struct ArrowDataTypeArbitraryOptions {
    pub allowed_dtypes: ArrowDataTypeArbitrarySelection,

    pub array_width_range: RangeInclusive<usize>,
    pub struct_num_fields_range: RangeInclusive<usize>,

    pub max_nesting_level: usize,
}

#[derive(Clone)]
pub struct ArrayArbitraryOptions {
    pub dtype: ArrowDataTypeArbitraryOptions,
}

impl Default for ArrowDataTypeArbitraryOptions {
    fn default() -> Self {
        Self {
            allowed_dtypes: ArrowDataTypeArbitrarySelection::all(),
            array_width_range: 0..=7,
            struct_num_fields_range: 0..=7,
            max_nesting_level: 5,
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for ArrayArbitraryOptions {
    fn default() -> Self {
        Self {
            dtype: Default::default(),
        }
    }
}

pub fn arrow_data_type_impl(
    options: Rc<ArrowDataTypeArbitraryOptions>,
    nesting_level: usize,
) -> impl Strategy<Value = ArrowDataType> {
    use ArrowDataTypeArbitrarySelection as S;
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
            _ if selection == S::NULL => Just(ArrowDataType::Null).boxed(),
            _ if selection == S::BOOLEAN => Just(ArrowDataType::Boolean).boxed(),
            _ if selection == S::INT8 => Just(ArrowDataType::Int8).boxed(),
            _ if selection == S::INT16 => Just(ArrowDataType::Int16).boxed(),
            _ if selection == S::INT32 => Just(ArrowDataType::Int32).boxed(),
            _ if selection == S::INT64 => Just(ArrowDataType::Int64).boxed(),
            _ if selection == S::INT128 => Just(ArrowDataType::Int128).boxed(),
            _ if selection == S::UINT8 => Just(ArrowDataType::UInt8).boxed(),
            _ if selection == S::UINT16 => Just(ArrowDataType::UInt16).boxed(),
            _ if selection == S::UINT32 => Just(ArrowDataType::UInt32).boxed(),
            _ if selection == S::UINT64 => Just(ArrowDataType::UInt64).boxed(),
            _ if selection == S::FLOAT32 => Just(ArrowDataType::Float32).boxed(),
            _ if selection == S::FLOAT64 => Just(ArrowDataType::Float64).boxed(),
            _ if selection == S::STRVIEW => Just(ArrowDataType::Utf8View).boxed(),
            _ if selection == S::BINVIEW => Just(ArrowDataType::BinaryView).boxed(),
            _ if selection == S::BINARY => Just(ArrowDataType::LargeBinary).boxed(),
            _ if selection == S::LIST => arrow_data_type_impl(options.clone(), nesting_level + 1)
                .prop_map(|dtype| {
                    let field = Field::new("item".into(), dtype, true);
                    ArrowDataType::LargeList(Box::new(field))
                })
                .boxed(),
            _ if selection == S::FIXED_SIZE_LIST => (
                arrow_data_type_impl(options.clone(), nesting_level + 1),
                options.array_width_range.clone(),
            )
                .prop_map(|(dtype, width)| {
                    let field = Field::new("item".into(), dtype, true);
                    ArrowDataType::FixedSizeList(Box::new(field), width)
                })
                .boxed(),
            _ if selection == S::STRUCT => proptest::collection::vec(
                arrow_data_type_impl(options.clone(), nesting_level + 1),
                options.struct_num_fields_range.clone(),
            )
            .prop_map(|dtypes| {
                let fields = dtypes
                    .into_iter()
                    .enumerate()
                    .map(|(i, dtype)| Field::new(format_pl_smallstr!("f{}", i + 1), dtype, true))
                    .collect();
                ArrowDataType::Struct(fields)
            })
            .boxed(),
            _ => unreachable!(),
        }
    })
}

pub fn arrow_data_type(
    options: ArrowDataTypeArbitraryOptions,
) -> impl Strategy<Value = ArrowDataType> {
    arrow_data_type_impl(Rc::new(options), 0)
}

pub fn array_with_dtype(
    dtype: ArrowDataType,
    size_range: impl Into<SizeRange>,
) -> impl Strategy<Value = Box<dyn Array>> {
    let size_range = size_range.into();
    match dtype {
        ArrowDataType::Null => null_array(size_range).prop_map(NullArray::boxed).boxed(),
        ArrowDataType::Boolean => boolean_array(size_range)
            .prop_map(BooleanArray::boxed)
            .boxed(),
        ArrowDataType::Int8 => primitive_array::<i8>(size_range)
            .prop_map(PrimitiveArray::boxed)
            .boxed(),
        ArrowDataType::Int16 => primitive_array::<i16>(size_range)
            .prop_map(PrimitiveArray::boxed)
            .boxed(),
        ArrowDataType::Int32 => primitive_array::<i32>(size_range)
            .prop_map(PrimitiveArray::boxed)
            .boxed(),
        ArrowDataType::Int64 => primitive_array::<i64>(size_range)
            .prop_map(PrimitiveArray::boxed)
            .boxed(),
        ArrowDataType::Int128 => primitive_array::<i128>(size_range)
            .prop_map(PrimitiveArray::boxed)
            .boxed(),
        ArrowDataType::UInt8 => primitive_array::<u8>(size_range)
            .prop_map(PrimitiveArray::boxed)
            .boxed(),
        ArrowDataType::UInt16 => primitive_array::<u16>(size_range)
            .prop_map(PrimitiveArray::boxed)
            .boxed(),
        ArrowDataType::UInt32 => primitive_array::<u32>(size_range)
            .prop_map(PrimitiveArray::boxed)
            .boxed(),
        ArrowDataType::UInt64 => primitive_array::<u64>(size_range)
            .prop_map(PrimitiveArray::boxed)
            .boxed(),
        ArrowDataType::Float32 => primitive_array::<f32>(size_range)
            .prop_map(PrimitiveArray::boxed)
            .boxed(),
        ArrowDataType::Float64 => primitive_array::<f64>(size_range)
            .prop_map(PrimitiveArray::boxed)
            .boxed(),
        ArrowDataType::LargeBinary => super::binary::proptest::binary_array(size_range)
            .prop_map(BinaryArray::boxed)
            .boxed(),
        ArrowDataType::FixedSizeList(field, width) => {
            super::fixed_size_list::proptest::fixed_size_list_array_with_dtype(
                size_range, field, width,
            )
            .prop_map(FixedSizeListArray::boxed)
            .boxed()
        },
        ArrowDataType::LargeList(field) => {
            super::list::proptest::list_array_with_dtype(size_range, field)
                .prop_map(ListArray::<i64>::boxed)
                .boxed()
        },
        ArrowDataType::Struct(fields) => {
            super::struct_::proptest::struct_array_with_fields(size_range, fields)
                .prop_map(StructArray::boxed)
                .boxed()
        },
        ArrowDataType::BinaryView => binview_array(size_range)
            .prop_map(BinaryViewArray::boxed)
            .boxed(),
        ArrowDataType::Utf8View => utf8view_array(size_range)
            .prop_map(Utf8ViewArray::boxed)
            .boxed(),
        ArrowDataType::Float16
        | ArrowDataType::Timestamp(..)
        | ArrowDataType::Date32
        | ArrowDataType::Date64
        | ArrowDataType::Time32(..)
        | ArrowDataType::Time64(..)
        | ArrowDataType::Duration(..)
        | ArrowDataType::Interval(..)
        | ArrowDataType::Binary
        | ArrowDataType::FixedSizeBinary(_)
        | ArrowDataType::Utf8
        | ArrowDataType::LargeUtf8
        | ArrowDataType::List(..)
        | ArrowDataType::Map(_, _)
        | ArrowDataType::Dictionary(..)
        | ArrowDataType::Decimal(..)
        | ArrowDataType::Decimal256(..)
        | ArrowDataType::Extension(..)
        | ArrowDataType::Unknown
        | ArrowDataType::Union(..) => unimplemented!(),
    }
}

pub fn array_with_options(
    size_range: impl Into<SizeRange>,
    options: ArrayArbitraryOptions,
) -> impl Strategy<Value = Box<dyn Array>> {
    let size_range = size_range.into();
    arrow_data_type(options.dtype)
        .prop_flat_map(move |dtype| array_with_dtype(dtype, size_range.clone()))
}

pub fn array(size_range: impl Into<SizeRange>) -> impl Strategy<Value = Box<dyn Array>> {
    let size_range = size_range.into();
    arrow_data_type(Default::default())
        .prop_flat_map(move |dtype| array_with_dtype(dtype, size_range.clone()))
}

pub fn null_array(size_range: impl Into<SizeRange>) -> impl Strategy<Value = NullArray> {
    let size_range = size_range.into();
    let (min, max) = size_range.start_end_incl();
    (min..=max).prop_map(|length| NullArray::new(ArrowDataType::Null, length))
}
