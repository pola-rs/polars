//! Defines different casting operators such as [`cast`] or [`primitive_to_binary`].

mod binary_to;
mod binview_to;
mod boolean_to;
mod decimal_to;
mod dictionary_to;
mod primitive_to;
mod utf8_to;

pub use binary_to::*;
#[cfg(feature = "dtype-decimal")]
pub use binview_to::binview_to_decimal;
use binview_to::binview_to_primitive_dyn;
pub use binview_to::utf8view_to_utf8;
pub use boolean_to::*;
pub use decimal_to::*;
pub use dictionary_to::*;
use polars_error::{polars_bail, polars_ensure, polars_err, PolarsResult};
use polars_utils::IdxSize;
pub use primitive_to::*;
pub use utf8_to::*;

use crate::array::*;
use crate::compute::cast::binview_to::{
    binview_to_dictionary, utf8view_to_date32_dyn, utf8view_to_dictionary,
    utf8view_to_naive_timestamp_dyn, view_to_binary,
};
use crate::datatypes::*;
use crate::match_integer_type;
use crate::offset::{Offset, Offsets};
use crate::temporal_conversions::utf8view_to_timestamp;

/// options defining how Cast kernels behave
#[derive(Clone, Copy, Debug, Default)]
pub struct CastOptionsImpl {
    /// default to false
    /// whether an overflowing cast should be converted to `None` (default), or be wrapped (i.e. `256i16 as u8 = 0` vectorized).
    /// Settings this to `true` is 5-6x faster for numeric types.
    pub wrapped: bool,
    /// default to false
    /// whether to cast to an integer at the best-effort
    pub partial: bool,
}

impl CastOptionsImpl {
    pub fn unchecked() -> Self {
        Self {
            wrapped: true,
            partial: false,
        }
    }
}

impl CastOptionsImpl {
    fn with_wrapped(&self, v: bool) -> Self {
        let mut option = *self;
        option.wrapped = v;
        option
    }
}

macro_rules! primitive_dyn {
    ($from:expr, $expr:tt) => {{
        let from = $from.as_any().downcast_ref().unwrap();
        Ok(Box::new($expr(from)))
    }};
    ($from:expr, $expr:tt, $to:expr) => {{
        let from = $from.as_any().downcast_ref().unwrap();
        Ok(Box::new($expr(from, $to)))
    }};
    ($from:expr, $expr:tt, $from_t:expr, $to:expr) => {{
        let from = $from.as_any().downcast_ref().unwrap();
        Ok(Box::new($expr(from, $from_t, $to)))
    }};
    ($from:expr, $expr:tt, $arg1:expr, $arg2:expr, $arg3:expr) => {{
        let from = $from.as_any().downcast_ref().unwrap();
        Ok(Box::new($expr(from, $arg1, $arg2, $arg3)))
    }};
}

fn cast_struct(
    array: &StructArray,
    to_type: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<StructArray> {
    let values = array.values();
    let fields = StructArray::get_fields(to_type);
    let new_values = values
        .iter()
        .zip(fields)
        .map(|(arr, field)| cast(arr.as_ref(), field.data_type(), options))
        .collect::<PolarsResult<Vec<_>>>()?;

    Ok(StructArray::new(
        to_type.clone(),
        new_values,
        array.validity().cloned(),
    ))
}

fn cast_list<O: Offset>(
    array: &ListArray<O>,
    to_type: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<ListArray<O>> {
    let values = array.values();
    let new_values = cast(
        values.as_ref(),
        ListArray::<O>::get_child_type(to_type),
        options,
    )?;

    Ok(ListArray::<O>::new(
        to_type.clone(),
        array.offsets().clone(),
        new_values,
        array.validity().cloned(),
    ))
}

fn cast_list_to_large_list(array: &ListArray<i32>, to_type: &ArrowDataType) -> ListArray<i64> {
    let offsets = array.offsets().into();

    ListArray::<i64>::new(
        to_type.clone(),
        offsets,
        array.values().clone(),
        array.validity().cloned(),
    )
}

fn cast_large_to_list(array: &ListArray<i64>, to_type: &ArrowDataType) -> ListArray<i32> {
    let offsets = array.offsets().try_into().expect("Convertme to error");

    ListArray::<i32>::new(
        to_type.clone(),
        offsets,
        array.values().clone(),
        array.validity().cloned(),
    )
}

fn cast_fixed_size_list_to_list<O: Offset>(
    fixed: &FixedSizeListArray,
    to_type: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<ListArray<O>> {
    let new_values = cast(
        fixed.values().as_ref(),
        ListArray::<O>::get_child_type(to_type),
        options,
    )?;

    let offsets = (0..=fixed.len())
        .map(|ix| O::from_as_usize(ix * fixed.size()))
        .collect::<Vec<_>>();
    // SAFETY: offsets _are_ monotonically increasing
    let offsets = unsafe { Offsets::new_unchecked(offsets) };

    Ok(ListArray::<O>::new(
        to_type.clone(),
        offsets.into(),
        new_values,
        fixed.validity().cloned(),
    ))
}

fn cast_list_to_fixed_size_list<O: Offset>(
    list: &ListArray<O>,
    inner: &Field,
    size: usize,
    options: CastOptionsImpl,
) -> PolarsResult<FixedSizeListArray> {
    let null_cnt = list.null_count();
    let new_values = if null_cnt == 0 {
        let offsets = list.offsets().buffer().iter();
        let expected =
            (list.offsets().first().to_usize()..list.len()).map(|ix| O::from_as_usize(ix * size));

        match offsets
            .zip(expected)
            .find(|(actual, expected)| *actual != expected)
        {
            Some(_) => polars_bail!(ComputeError:
                "not all elements have the specified width {size}"
            ),
            None => {
                let sliced_values = list.values().sliced(
                    list.offsets().first().to_usize(),
                    list.offsets().range().to_usize(),
                );
                cast(sliced_values.as_ref(), inner.data_type(), options)?
            },
        }
    } else {
        let offsets = list.offsets().as_slice();
        // Check the lengths of each list are equal to the fixed size.
        // SAFETY: we know the index is in bound.
        let mut expected_offset = unsafe { *offsets.get_unchecked(0) } + O::from_as_usize(size);
        for i in 1..=list.len() {
            // SAFETY: we know the index is in bound.
            let current_offset = unsafe { *offsets.get_unchecked(i) };
            if list.is_null(i - 1) {
                expected_offset = current_offset + O::from_as_usize(size);
            } else {
                polars_ensure!(current_offset == expected_offset, ComputeError:
            "not all elements have the specified width {size}");
                expected_offset += O::from_as_usize(size);
            }
        }

        // Build take indices for the values. This is used to fill in the null slots.
        let mut indices =
            MutablePrimitiveArray::<IdxSize>::with_capacity(list.values().len() + null_cnt * size);
        for i in 0..list.len() {
            if list.is_null(i) {
                indices.extend_constant(size, None)
            } else {
                // SAFETY: we know the index is in bound.
                let current_offset = unsafe { *offsets.get_unchecked(i) };
                for j in 0..size {
                    indices.push(Some(
                        (current_offset + O::from_as_usize(j)).to_usize() as IdxSize
                    ));
                }
            }
        }
        let take_values = unsafe {
            crate::compute::take::take_unchecked(list.values().as_ref(), &indices.freeze())
        };

        cast(take_values.as_ref(), inner.data_type(), options)?
    };
    FixedSizeListArray::try_new(
        ArrowDataType::FixedSizeList(Box::new(inner.clone()), size),
        new_values,
        list.validity().cloned(),
    )
    .map_err(|_| polars_err!(ComputeError: "not all elements have the specified width {size}"))
}

pub fn cast_default(array: &dyn Array, to_type: &ArrowDataType) -> PolarsResult<Box<dyn Array>> {
    cast(array, to_type, Default::default())
}

pub fn cast_unchecked(array: &dyn Array, to_type: &ArrowDataType) -> PolarsResult<Box<dyn Array>> {
    cast(array, to_type, CastOptionsImpl::unchecked())
}

/// Cast `array` to the provided data type and return a new [`Array`] with
/// type `to_type`, if possible.
///
/// Behavior:
/// * PrimitiveArray to PrimitiveArray: overflowing cast will be None
/// * Boolean to Utf8: `true` => '1', `false` => `0`
/// * Utf8 to numeric: strings that can't be parsed to numbers return null, float strings
///   in integer casts return null
/// * Numeric to boolean: 0 returns `false`, any other value returns `true`
/// * List to List: the underlying data type is cast
/// * Fixed Size List to List: the underlying data type is cast
/// * List to Fixed Size List: the offsets are checked for valid order, then the
///   underlying type is cast.
/// * Struct to Struct: the underlying fields are cast.
/// * PrimitiveArray to List: a list array with 1 value per slot is created
/// * Date32 and Date64: precision lost when going to higher interval
/// * Time32 and Time64: precision lost when going to higher interval
/// * Timestamp and Date{32|64}: precision lost when going to higher interval
/// * Temporal to/from backing primitive: zero-copy with data type change
///
/// Unsupported Casts
/// * non-`StructArray` to `StructArray` or `StructArray` to non-`StructArray`
/// * List to primitive
/// * Utf8 to boolean
/// * Interval and duration
pub fn cast(
    array: &dyn Array,
    to_type: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn Array>> {
    use ArrowDataType::*;
    let from_type = array.data_type();

    // clone array if types are the same
    if from_type == to_type {
        return Ok(clone(array));
    }

    let as_options = options.with_wrapped(true);
    match (from_type, to_type) {
        (Null, _) | (_, Null) => Ok(new_null_array(to_type.clone(), array.len())),
        (Struct(from_fd), Struct(to_fd)) => {
            polars_ensure!(from_fd.len() == to_fd.len(), InvalidOperation: "Cannot cast struct with different number of fields.");
            cast_struct(array.as_any().downcast_ref().unwrap(), to_type, options).map(|x| x.boxed())
        },
        (Struct(_), _) | (_, Struct(_)) => polars_bail!(InvalidOperation:
            "Cannot cast from struct to other types"
        ),
        (Dictionary(index_type, ..), _) => match_integer_type!(index_type, |$T| {
            dictionary_cast_dyn::<$T>(array, to_type, options)
        }),
        (_, Dictionary(index_type, value_type, _)) => match_integer_type!(index_type, |$T| {
            cast_to_dictionary::<$T>(array, value_type, options)
        }),
        // not supported by polars
        // (List(_), FixedSizeList(inner, size)) => cast_list_to_fixed_size_list::<i32>(
        //     array.as_any().downcast_ref().unwrap(),
        //     inner.as_ref(),
        //     *size,
        //     options,
        // )
        // .map(|x| x.boxed()),
        (LargeList(_), FixedSizeList(inner, size)) => cast_list_to_fixed_size_list::<i64>(
            array.as_any().downcast_ref().unwrap(),
            inner.as_ref(),
            *size,
            options,
        )
        .map(|x| x.boxed()),
        (FixedSizeList(_, _), List(_)) => cast_fixed_size_list_to_list::<i32>(
            array.as_any().downcast_ref().unwrap(),
            to_type,
            options,
        )
        .map(|x| x.boxed()),
        (FixedSizeList(_, _), LargeList(_)) => cast_fixed_size_list_to_list::<i64>(
            array.as_any().downcast_ref().unwrap(),
            to_type,
            options,
        )
        .map(|x| x.boxed()),
        (BinaryView, _) => match to_type {
            Utf8View => array
                .as_any()
                .downcast_ref::<BinaryViewArray>()
                .unwrap()
                .to_utf8view()
                .map(|arr| arr.boxed()),
            LargeBinary => Ok(binview_to::view_to_binary::<i64>(
                array.as_any().downcast_ref().unwrap(),
            )
            .boxed()),
            UInt8 => binview_to_primitive_dyn::<u8>(array, to_type, options),
            UInt16 => binview_to_primitive_dyn::<u16>(array, to_type, options),
            UInt32 => binview_to_primitive_dyn::<u32>(array, to_type, options),
            UInt64 => binview_to_primitive_dyn::<u64>(array, to_type, options),
            Int8 => binview_to_primitive_dyn::<i8>(array, to_type, options),
            Int16 => binview_to_primitive_dyn::<i16>(array, to_type, options),
            Int32 => binview_to_primitive_dyn::<i32>(array, to_type, options),
            Int64 => binview_to_primitive_dyn::<i64>(array, to_type, options),
            Float32 => binview_to_primitive_dyn::<f32>(array, to_type, options),
            Float64 => binview_to_primitive_dyn::<f64>(array, to_type, options),
            LargeList(inner) if matches!(inner.data_type, ArrowDataType::UInt8) => {
                let bin_array = view_to_binary::<i64>(array.as_any().downcast_ref().unwrap());
                Ok(binary_to_list(&bin_array, to_type.clone()).boxed())
            },
            _ => polars_bail!(InvalidOperation:
                "casting from {from_type:?} to {to_type:?} not supported",
            ),
        },
        (LargeList(_), LargeList(_)) => {
            cast_list::<i64>(array.as_any().downcast_ref().unwrap(), to_type, options)
                .map(|x| x.boxed())
        },
        (List(lhs), LargeList(rhs)) if lhs == rhs => {
            Ok(cast_list_to_large_list(array.as_any().downcast_ref().unwrap(), to_type).boxed())
        },
        (LargeList(lhs), List(rhs)) if lhs == rhs => {
            Ok(cast_large_to_list(array.as_any().downcast_ref().unwrap(), to_type).boxed())
        },

        (_, List(to)) => {
            // cast primitive to list's primitive
            let values = cast(array, &to.data_type, options)?;
            // create offsets, where if array.len() = 2, we have [0,1,2]
            let offsets = (0..=array.len() as i32).collect::<Vec<_>>();
            // SAFETY: offsets _are_ monotonically increasing
            let offsets = unsafe { Offsets::new_unchecked(offsets) };

            let list_array = ListArray::<i32>::new(to_type.clone(), offsets.into(), values, None);

            Ok(Box::new(list_array))
        },

        (_, LargeList(to)) if from_type != &LargeBinary => {
            // cast primitive to list's primitive
            let values = cast(array, &to.data_type, options)?;
            // create offsets, where if array.len() = 2, we have [0,1,2]
            let offsets = (0..=array.len() as i64).collect::<Vec<_>>();
            // SAFETY: offsets _are_ monotonically increasing
            let offsets = unsafe { Offsets::new_unchecked(offsets) };

            let list_array = ListArray::<i64>::new(
                to_type.clone(),
                offsets.into(),
                values,
                array.validity().cloned(),
            );

            Ok(Box::new(list_array))
        },

        (Utf8View, _) => {
            let arr = array.as_any().downcast_ref::<Utf8ViewArray>().unwrap();

            match to_type {
                BinaryView => Ok(arr.to_binview().boxed()),
                LargeUtf8 => Ok(binview_to::utf8view_to_utf8::<i64>(arr).boxed()),
                UInt8
                | UInt16
                | UInt32
                | UInt64
                | Int8
                | Int16
                | Int32
                | Int64
                | Float32
                | Float64
                | Decimal(_, _) => cast(&arr.to_binview(), to_type, options),
                Timestamp(time_unit, None) => {
                    utf8view_to_naive_timestamp_dyn(array, time_unit.to_owned())
                },
                Timestamp(time_unit, Some(time_zone)) => utf8view_to_timestamp(
                    array.as_any().downcast_ref().unwrap(),
                    RFC3339,
                    time_zone.clone(),
                    time_unit.to_owned(),
                )
                .map(|arr| arr.boxed()),
                Date32 => utf8view_to_date32_dyn(array),
                _ => polars_bail!(InvalidOperation:
                    "casting from {from_type:?} to {to_type:?} not supported",
                ),
            }
        },

        (_, Boolean) => match from_type {
            UInt8 => primitive_to_boolean_dyn::<u8>(array, to_type.clone()),
            UInt16 => primitive_to_boolean_dyn::<u16>(array, to_type.clone()),
            UInt32 => primitive_to_boolean_dyn::<u32>(array, to_type.clone()),
            UInt64 => primitive_to_boolean_dyn::<u64>(array, to_type.clone()),
            Int8 => primitive_to_boolean_dyn::<i8>(array, to_type.clone()),
            Int16 => primitive_to_boolean_dyn::<i16>(array, to_type.clone()),
            Int32 => primitive_to_boolean_dyn::<i32>(array, to_type.clone()),
            Int64 => primitive_to_boolean_dyn::<i64>(array, to_type.clone()),
            Float32 => primitive_to_boolean_dyn::<f32>(array, to_type.clone()),
            Float64 => primitive_to_boolean_dyn::<f64>(array, to_type.clone()),
            Decimal(_, _) => primitive_to_boolean_dyn::<i128>(array, to_type.clone()),
            _ => polars_bail!(InvalidOperation:
                "casting from {from_type:?} to {to_type:?} not supported",
            ),
        },
        (Boolean, _) => match to_type {
            UInt8 => boolean_to_primitive_dyn::<u8>(array),
            UInt16 => boolean_to_primitive_dyn::<u16>(array),
            UInt32 => boolean_to_primitive_dyn::<u32>(array),
            UInt64 => boolean_to_primitive_dyn::<u64>(array),
            Int8 => boolean_to_primitive_dyn::<i8>(array),
            Int16 => boolean_to_primitive_dyn::<i16>(array),
            Int32 => boolean_to_primitive_dyn::<i32>(array),
            Int64 => boolean_to_primitive_dyn::<i64>(array),
            Float32 => boolean_to_primitive_dyn::<f32>(array),
            Float64 => boolean_to_primitive_dyn::<f64>(array),
            Utf8View => boolean_to_utf8view_dyn(array),
            BinaryView => boolean_to_binaryview_dyn(array),
            _ => polars_bail!(InvalidOperation:
                "casting from {from_type:?} to {to_type:?} not supported",
            ),
        },
        (_, BinaryView) => from_to_binview(array, from_type, to_type).map(|arr| arr.boxed()),
        (_, Utf8View) => match from_type {
            LargeUtf8 => Ok(utf8_to_utf8view(
                array.as_any().downcast_ref::<Utf8Array<i64>>().unwrap(),
            )
            .boxed()),
            Utf8 => Ok(
                utf8_to_utf8view(array.as_any().downcast_ref::<Utf8Array<i32>>().unwrap()).boxed(),
            ),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => Ok(decimal_to_utf8view_dyn(array).boxed()),
            _ => from_to_binview(array, from_type, to_type)
                .map(|arr| unsafe { arr.to_utf8view_unchecked() }.boxed()),
        },
        (Utf8, _) => match to_type {
            LargeUtf8 => Ok(Box::new(utf8_to_large_utf8(
                array.as_any().downcast_ref().unwrap(),
            ))),
            _ => polars_bail!(InvalidOperation:
                "casting from {from_type:?} to {to_type:?} not supported",
            ),
        },
        (LargeUtf8, _) => match to_type {
            LargeBinary => Ok(utf8_to_binary::<i64>(
                array.as_any().downcast_ref().unwrap(),
                to_type.clone(),
            )
            .boxed()),
            _ => polars_bail!(InvalidOperation:
                "casting from {from_type:?} to {to_type:?} not supported",
            ),
        },
        (_, LargeUtf8) => match from_type {
            UInt8 => primitive_to_utf8_dyn::<u8, i64>(array),
            LargeBinary => {
                binary_to_utf8::<i64>(array.as_any().downcast_ref().unwrap(), to_type.clone())
                    .map(|x| x.boxed())
            },
            _ => polars_bail!(InvalidOperation:
                "casting from {from_type:?} to {to_type:?} not supported",
            ),
        },

        (Binary, _) => match to_type {
            LargeBinary => Ok(Box::new(binary_to_large_binary(
                array.as_any().downcast_ref().unwrap(),
                to_type.clone(),
            ))),
            _ => polars_bail!(InvalidOperation:
                "casting from {from_type:?} to {to_type:?} not supported",
            ),
        },

        (LargeBinary, _) => match to_type {
            UInt8 => binary_to_primitive_dyn::<i64, u8>(array, to_type, options),
            UInt16 => binary_to_primitive_dyn::<i64, u16>(array, to_type, options),
            UInt32 => binary_to_primitive_dyn::<i64, u32>(array, to_type, options),
            UInt64 => binary_to_primitive_dyn::<i64, u64>(array, to_type, options),
            Int8 => binary_to_primitive_dyn::<i64, i8>(array, to_type, options),
            Int16 => binary_to_primitive_dyn::<i64, i16>(array, to_type, options),
            Int32 => binary_to_primitive_dyn::<i64, i32>(array, to_type, options),
            Int64 => binary_to_primitive_dyn::<i64, i64>(array, to_type, options),
            Float32 => binary_to_primitive_dyn::<i64, f32>(array, to_type, options),
            Float64 => binary_to_primitive_dyn::<i64, f64>(array, to_type, options),
            Binary => {
                binary_large_to_binary(array.as_any().downcast_ref().unwrap(), to_type.clone())
                    .map(|x| x.boxed())
            },
            LargeUtf8 => {
                binary_to_utf8::<i64>(array.as_any().downcast_ref().unwrap(), to_type.clone())
                    .map(|x| x.boxed())
            },
            _ => polars_bail!(InvalidOperation:
                "casting from {from_type:?} to {to_type:?} not supported",
            ),
        },
        (FixedSizeBinary(_), _) => match to_type {
            Binary => Ok(fixed_size_binary_binary::<i32>(
                array.as_any().downcast_ref().unwrap(),
                to_type.clone(),
            )
            .boxed()),
            LargeBinary => Ok(fixed_size_binary_binary::<i64>(
                array.as_any().downcast_ref().unwrap(),
                to_type.clone(),
            )
            .boxed()),
            _ => polars_bail!(InvalidOperation:
                "casting from {from_type:?} to {to_type:?} not supported",
            ),
        },
        // start numeric casts
        (UInt8, UInt16) => primitive_to_primitive_dyn::<u8, u16>(array, to_type, as_options),
        (UInt8, UInt32) => primitive_to_primitive_dyn::<u8, u32>(array, to_type, as_options),
        (UInt8, UInt64) => primitive_to_primitive_dyn::<u8, u64>(array, to_type, as_options),
        (UInt8, Int8) => primitive_to_primitive_dyn::<u8, i8>(array, to_type, options),
        (UInt8, Int16) => primitive_to_primitive_dyn::<u8, i16>(array, to_type, options),
        (UInt8, Int32) => primitive_to_primitive_dyn::<u8, i32>(array, to_type, options),
        (UInt8, Int64) => primitive_to_primitive_dyn::<u8, i64>(array, to_type, options),
        (UInt8, Float32) => primitive_to_primitive_dyn::<u8, f32>(array, to_type, as_options),
        (UInt8, Float64) => primitive_to_primitive_dyn::<u8, f64>(array, to_type, as_options),
        (UInt8, Decimal(p, s)) => integer_to_decimal_dyn::<u8>(array, *p, *s),

        (UInt16, UInt8) => primitive_to_primitive_dyn::<u16, u8>(array, to_type, options),
        (UInt16, UInt32) => primitive_to_primitive_dyn::<u16, u32>(array, to_type, as_options),
        (UInt16, UInt64) => primitive_to_primitive_dyn::<u16, u64>(array, to_type, as_options),
        (UInt16, Int8) => primitive_to_primitive_dyn::<u16, i8>(array, to_type, options),
        (UInt16, Int16) => primitive_to_primitive_dyn::<u16, i16>(array, to_type, options),
        (UInt16, Int32) => primitive_to_primitive_dyn::<u16, i32>(array, to_type, options),
        (UInt16, Int64) => primitive_to_primitive_dyn::<u16, i64>(array, to_type, options),
        (UInt16, Float32) => primitive_to_primitive_dyn::<u16, f32>(array, to_type, as_options),
        (UInt16, Float64) => primitive_to_primitive_dyn::<u16, f64>(array, to_type, as_options),
        (UInt16, Decimal(p, s)) => integer_to_decimal_dyn::<u16>(array, *p, *s),

        (UInt32, UInt8) => primitive_to_primitive_dyn::<u32, u8>(array, to_type, options),
        (UInt32, UInt16) => primitive_to_primitive_dyn::<u32, u16>(array, to_type, options),
        (UInt32, UInt64) => primitive_to_primitive_dyn::<u32, u64>(array, to_type, as_options),
        (UInt32, Int8) => primitive_to_primitive_dyn::<u32, i8>(array, to_type, options),
        (UInt32, Int16) => primitive_to_primitive_dyn::<u32, i16>(array, to_type, options),
        (UInt32, Int32) => primitive_to_primitive_dyn::<u32, i32>(array, to_type, options),
        (UInt32, Int64) => primitive_to_primitive_dyn::<u32, i64>(array, to_type, options),
        (UInt32, Float32) => primitive_to_primitive_dyn::<u32, f32>(array, to_type, as_options),
        (UInt32, Float64) => primitive_to_primitive_dyn::<u32, f64>(array, to_type, as_options),
        (UInt32, Decimal(p, s)) => integer_to_decimal_dyn::<u32>(array, *p, *s),

        (UInt64, UInt8) => primitive_to_primitive_dyn::<u64, u8>(array, to_type, options),
        (UInt64, UInt16) => primitive_to_primitive_dyn::<u64, u16>(array, to_type, options),
        (UInt64, UInt32) => primitive_to_primitive_dyn::<u64, u32>(array, to_type, options),
        (UInt64, Int8) => primitive_to_primitive_dyn::<u64, i8>(array, to_type, options),
        (UInt64, Int16) => primitive_to_primitive_dyn::<u64, i16>(array, to_type, options),
        (UInt64, Int32) => primitive_to_primitive_dyn::<u64, i32>(array, to_type, options),
        (UInt64, Int64) => primitive_to_primitive_dyn::<u64, i64>(array, to_type, options),
        (UInt64, Float32) => primitive_to_primitive_dyn::<u64, f32>(array, to_type, as_options),
        (UInt64, Float64) => primitive_to_primitive_dyn::<u64, f64>(array, to_type, as_options),
        (UInt64, Decimal(p, s)) => integer_to_decimal_dyn::<u64>(array, *p, *s),

        (Int8, UInt8) => primitive_to_primitive_dyn::<i8, u8>(array, to_type, options),
        (Int8, UInt16) => primitive_to_primitive_dyn::<i8, u16>(array, to_type, options),
        (Int8, UInt32) => primitive_to_primitive_dyn::<i8, u32>(array, to_type, options),
        (Int8, UInt64) => primitive_to_primitive_dyn::<i8, u64>(array, to_type, options),
        (Int8, Int16) => primitive_to_primitive_dyn::<i8, i16>(array, to_type, as_options),
        (Int8, Int32) => primitive_to_primitive_dyn::<i8, i32>(array, to_type, as_options),
        (Int8, Int64) => primitive_to_primitive_dyn::<i8, i64>(array, to_type, as_options),
        (Int8, Float32) => primitive_to_primitive_dyn::<i8, f32>(array, to_type, as_options),
        (Int8, Float64) => primitive_to_primitive_dyn::<i8, f64>(array, to_type, as_options),
        (Int8, Decimal(p, s)) => integer_to_decimal_dyn::<i8>(array, *p, *s),

        (Int16, UInt8) => primitive_to_primitive_dyn::<i16, u8>(array, to_type, options),
        (Int16, UInt16) => primitive_to_primitive_dyn::<i16, u16>(array, to_type, options),
        (Int16, UInt32) => primitive_to_primitive_dyn::<i16, u32>(array, to_type, options),
        (Int16, UInt64) => primitive_to_primitive_dyn::<i16, u64>(array, to_type, options),
        (Int16, Int8) => primitive_to_primitive_dyn::<i16, i8>(array, to_type, options),
        (Int16, Int32) => primitive_to_primitive_dyn::<i16, i32>(array, to_type, as_options),
        (Int16, Int64) => primitive_to_primitive_dyn::<i16, i64>(array, to_type, as_options),
        (Int16, Float32) => primitive_to_primitive_dyn::<i16, f32>(array, to_type, as_options),
        (Int16, Float64) => primitive_to_primitive_dyn::<i16, f64>(array, to_type, as_options),
        (Int16, Decimal(p, s)) => integer_to_decimal_dyn::<i16>(array, *p, *s),

        (Int32, UInt8) => primitive_to_primitive_dyn::<i32, u8>(array, to_type, options),
        (Int32, UInt16) => primitive_to_primitive_dyn::<i32, u16>(array, to_type, options),
        (Int32, UInt32) => primitive_to_primitive_dyn::<i32, u32>(array, to_type, options),
        (Int32, UInt64) => primitive_to_primitive_dyn::<i32, u64>(array, to_type, options),
        (Int32, Int8) => primitive_to_primitive_dyn::<i32, i8>(array, to_type, options),
        (Int32, Int16) => primitive_to_primitive_dyn::<i32, i16>(array, to_type, options),
        (Int32, Int64) => primitive_to_primitive_dyn::<i32, i64>(array, to_type, as_options),
        (Int32, Float32) => primitive_to_primitive_dyn::<i32, f32>(array, to_type, as_options),
        (Int32, Float64) => primitive_to_primitive_dyn::<i32, f64>(array, to_type, as_options),
        (Int32, Decimal(p, s)) => integer_to_decimal_dyn::<i32>(array, *p, *s),

        (Int64, UInt8) => primitive_to_primitive_dyn::<i64, u8>(array, to_type, options),
        (Int64, UInt16) => primitive_to_primitive_dyn::<i64, u16>(array, to_type, options),
        (Int64, UInt32) => primitive_to_primitive_dyn::<i64, u32>(array, to_type, options),
        (Int64, UInt64) => primitive_to_primitive_dyn::<i64, u64>(array, to_type, options),
        (Int64, Int8) => primitive_to_primitive_dyn::<i64, i8>(array, to_type, options),
        (Int64, Int16) => primitive_to_primitive_dyn::<i64, i16>(array, to_type, options),
        (Int64, Int32) => primitive_to_primitive_dyn::<i64, i32>(array, to_type, options),
        (Int64, Float32) => primitive_to_primitive_dyn::<i64, f32>(array, to_type, options),
        (Int64, Float64) => primitive_to_primitive_dyn::<i64, f64>(array, to_type, as_options),
        (Int64, Decimal(p, s)) => integer_to_decimal_dyn::<i64>(array, *p, *s),

        (Float16, Float32) => {
            let from = array.as_any().downcast_ref().unwrap();
            Ok(f16_to_f32(from).boxed())
        },

        (Float32, UInt8) => primitive_to_primitive_dyn::<f32, u8>(array, to_type, options),
        (Float32, UInt16) => primitive_to_primitive_dyn::<f32, u16>(array, to_type, options),
        (Float32, UInt32) => primitive_to_primitive_dyn::<f32, u32>(array, to_type, options),
        (Float32, UInt64) => primitive_to_primitive_dyn::<f32, u64>(array, to_type, options),
        (Float32, Int8) => primitive_to_primitive_dyn::<f32, i8>(array, to_type, options),
        (Float32, Int16) => primitive_to_primitive_dyn::<f32, i16>(array, to_type, options),
        (Float32, Int32) => primitive_to_primitive_dyn::<f32, i32>(array, to_type, options),
        (Float32, Int64) => primitive_to_primitive_dyn::<f32, i64>(array, to_type, options),
        (Float32, Float64) => primitive_to_primitive_dyn::<f32, f64>(array, to_type, as_options),
        (Float32, Decimal(p, s)) => float_to_decimal_dyn::<f32>(array, *p, *s),

        (Float64, UInt8) => primitive_to_primitive_dyn::<f64, u8>(array, to_type, options),
        (Float64, UInt16) => primitive_to_primitive_dyn::<f64, u16>(array, to_type, options),
        (Float64, UInt32) => primitive_to_primitive_dyn::<f64, u32>(array, to_type, options),
        (Float64, UInt64) => primitive_to_primitive_dyn::<f64, u64>(array, to_type, options),
        (Float64, Int8) => primitive_to_primitive_dyn::<f64, i8>(array, to_type, options),
        (Float64, Int16) => primitive_to_primitive_dyn::<f64, i16>(array, to_type, options),
        (Float64, Int32) => primitive_to_primitive_dyn::<f64, i32>(array, to_type, options),
        (Float64, Int64) => primitive_to_primitive_dyn::<f64, i64>(array, to_type, options),
        (Float64, Float32) => primitive_to_primitive_dyn::<f64, f32>(array, to_type, options),
        (Float64, Decimal(p, s)) => float_to_decimal_dyn::<f64>(array, *p, *s),

        (Decimal(_, _), UInt8) => decimal_to_integer_dyn::<u8>(array),
        (Decimal(_, _), UInt16) => decimal_to_integer_dyn::<u16>(array),
        (Decimal(_, _), UInt32) => decimal_to_integer_dyn::<u32>(array),
        (Decimal(_, _), UInt64) => decimal_to_integer_dyn::<u64>(array),
        (Decimal(_, _), Int8) => decimal_to_integer_dyn::<i8>(array),
        (Decimal(_, _), Int16) => decimal_to_integer_dyn::<i16>(array),
        (Decimal(_, _), Int32) => decimal_to_integer_dyn::<i32>(array),
        (Decimal(_, _), Int64) => decimal_to_integer_dyn::<i64>(array),
        (Decimal(_, _), Float32) => decimal_to_float_dyn::<f32>(array),
        (Decimal(_, _), Float64) => decimal_to_float_dyn::<f64>(array),
        (Decimal(_, _), Decimal(to_p, to_s)) => decimal_to_decimal_dyn(array, *to_p, *to_s),
        // end numeric casts

        // temporal casts
        (Int32, Date32) => primitive_to_same_primitive_dyn::<i32>(array, to_type),
        (Int32, Time32(TimeUnit::Second)) => primitive_to_same_primitive_dyn::<i32>(array, to_type),
        (Int32, Time32(TimeUnit::Millisecond)) => {
            primitive_to_same_primitive_dyn::<i32>(array, to_type)
        },
        // No support for microsecond/nanosecond with i32
        (Date32, Int32) => primitive_to_same_primitive_dyn::<i32>(array, to_type),
        (Date32, Int64) => primitive_to_primitive_dyn::<i32, i64>(array, to_type, options),
        (Time32(_), Int32) => primitive_to_same_primitive_dyn::<i32>(array, to_type),
        (Int64, Date64) => primitive_to_same_primitive_dyn::<i64>(array, to_type),
        // No support for second/milliseconds with i64
        (Int64, Time64(TimeUnit::Microsecond)) => {
            primitive_to_same_primitive_dyn::<i64>(array, to_type)
        },
        (Int64, Time64(TimeUnit::Nanosecond)) => {
            primitive_to_same_primitive_dyn::<i64>(array, to_type)
        },

        (Date64, Int32) => primitive_to_primitive_dyn::<i64, i32>(array, to_type, options),
        (Date64, Int64) => primitive_to_same_primitive_dyn::<i64>(array, to_type),
        (Time64(_), Int64) => primitive_to_same_primitive_dyn::<i64>(array, to_type),
        (Date32, Date64) => primitive_dyn!(array, date32_to_date64),
        (Date64, Date32) => primitive_dyn!(array, date64_to_date32),
        (Time32(TimeUnit::Second), Time32(TimeUnit::Millisecond)) => {
            primitive_dyn!(array, time32s_to_time32ms)
        },
        (Time32(TimeUnit::Millisecond), Time32(TimeUnit::Second)) => {
            primitive_dyn!(array, time32ms_to_time32s)
        },
        (Time32(from_unit), Time64(to_unit)) => {
            primitive_dyn!(array, time32_to_time64, *from_unit, *to_unit)
        },
        (Time64(TimeUnit::Microsecond), Time64(TimeUnit::Nanosecond)) => {
            primitive_dyn!(array, time64us_to_time64ns)
        },
        (Time64(TimeUnit::Nanosecond), Time64(TimeUnit::Microsecond)) => {
            primitive_dyn!(array, time64ns_to_time64us)
        },
        (Time64(from_unit), Time32(to_unit)) => {
            primitive_dyn!(array, time64_to_time32, *from_unit, *to_unit)
        },
        (Timestamp(_, _), Int64) => primitive_to_same_primitive_dyn::<i64>(array, to_type),
        (Int64, Timestamp(_, _)) => primitive_to_same_primitive_dyn::<i64>(array, to_type),
        (Timestamp(from_unit, _), Timestamp(to_unit, tz)) => {
            primitive_dyn!(array, timestamp_to_timestamp, *from_unit, *to_unit, tz)
        },
        (Timestamp(from_unit, _), Date32) => primitive_dyn!(array, timestamp_to_date32, *from_unit),
        (Timestamp(from_unit, _), Date64) => primitive_dyn!(array, timestamp_to_date64, *from_unit),

        (Int64, Duration(_)) => primitive_to_same_primitive_dyn::<i64>(array, to_type),
        (Duration(_), Int64) => primitive_to_same_primitive_dyn::<i64>(array, to_type),

        // Not supported by Polars.
        // (Interval(IntervalUnit::DayTime), Interval(IntervalUnit::MonthDayNano)) => {
        //     primitive_dyn!(array, days_ms_to_months_days_ns)
        // },
        // (Interval(IntervalUnit::YearMonth), Interval(IntervalUnit::MonthDayNano)) => {
        //     primitive_dyn!(array, months_to_months_days_ns)
        // },
        _ => polars_bail!(InvalidOperation:
            "casting from {from_type:?} to {to_type:?} not supported",
        ),
    }
}

/// Attempts to encode an array into an `ArrayDictionary` with index
/// type K and value (dictionary) type value_type
///
/// K is the key type
fn cast_to_dictionary<K: DictionaryKey>(
    array: &dyn Array,
    dict_value_type: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn Array>> {
    let array = cast(array, dict_value_type, options)?;
    let array = array.as_ref();
    match *dict_value_type {
        ArrowDataType::Int8 => primitive_to_dictionary_dyn::<i8, K>(array),
        ArrowDataType::Int16 => primitive_to_dictionary_dyn::<i16, K>(array),
        ArrowDataType::Int32 => primitive_to_dictionary_dyn::<i32, K>(array),
        ArrowDataType::Int64 => primitive_to_dictionary_dyn::<i64, K>(array),
        ArrowDataType::UInt8 => primitive_to_dictionary_dyn::<u8, K>(array),
        ArrowDataType::UInt16 => primitive_to_dictionary_dyn::<u16, K>(array),
        ArrowDataType::UInt32 => primitive_to_dictionary_dyn::<u32, K>(array),
        ArrowDataType::UInt64 => primitive_to_dictionary_dyn::<u64, K>(array),
        ArrowDataType::BinaryView => {
            binview_to_dictionary::<K>(array.as_any().downcast_ref().unwrap())
                .map(|arr| arr.boxed())
        },
        ArrowDataType::Utf8View => {
            utf8view_to_dictionary::<K>(array.as_any().downcast_ref().unwrap())
                .map(|arr| arr.boxed())
        },
        ArrowDataType::LargeUtf8 => utf8_to_dictionary_dyn::<i64, K>(array),
        ArrowDataType::LargeBinary => binary_to_dictionary_dyn::<i64, K>(array),
        ArrowDataType::Time64(_) => primitive_to_dictionary_dyn::<i64, K>(array),
        ArrowDataType::Timestamp(_, _) => primitive_to_dictionary_dyn::<i64, K>(array),
        ArrowDataType::Date32 => primitive_to_dictionary_dyn::<i32, K>(array),
        _ => polars_bail!(ComputeError:
            "unsupported output type for dictionary packing: {dict_value_type:?}"
        ),
    }
}

fn from_to_binview(
    array: &dyn Array,
    from_type: &ArrowDataType,
    to_type: &ArrowDataType,
) -> PolarsResult<BinaryViewArray> {
    use ArrowDataType::*;
    let binview = match from_type {
        UInt8 => primitive_to_binview_dyn::<u8>(array),
        UInt16 => primitive_to_binview_dyn::<u16>(array),
        UInt32 => primitive_to_binview_dyn::<u32>(array),
        UInt64 => primitive_to_binview_dyn::<u64>(array),
        Int8 => primitive_to_binview_dyn::<i8>(array),
        Int16 => primitive_to_binview_dyn::<i16>(array),
        Int32 => primitive_to_binview_dyn::<i32>(array),
        Int64 => primitive_to_binview_dyn::<i64>(array),
        Float32 => primitive_to_binview_dyn::<f32>(array),
        Float64 => primitive_to_binview_dyn::<f64>(array),
        Binary => binary_to_binview::<i32>(array.as_any().downcast_ref().unwrap()),
        FixedSizeBinary(_) => fixed_size_binary_to_binview(array.as_any().downcast_ref().unwrap()),
        LargeBinary => binary_to_binview::<i64>(array.as_any().downcast_ref().unwrap()),
        _ => polars_bail!(InvalidOperation:
            "casting from {from_type:?} to {to_type:?} not supported",
        ),
    };
    Ok(binview)
}
