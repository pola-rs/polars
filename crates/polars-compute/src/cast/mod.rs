//! Defines different casting operators such as [`cast`] or [`primitive_to_binary`].

mod binary_to;
mod binview_to;
mod boolean_to;
#[cfg(feature = "dtype-decimal")]
mod decimal_to;
mod dictionary_to;
mod primitive_to;
mod utf8_to;

use arrow::bitmap::MutableBitmap;
pub use binary_to::*;
#[cfg(feature = "dtype-decimal")]
pub use binview_to::binview_to_decimal;
use binview_to::utf8view_to_primitive_dyn;
pub use binview_to::utf8view_to_utf8;
pub use boolean_to::*;
#[cfg(feature = "dtype-decimal")]
pub use decimal_to::*;
pub mod temporal;
use arrow::array::*;
use arrow::datatypes::*;
use arrow::match_integer_type;
use arrow::offset::{Offset, Offsets};
use binview_to::{
    binview_to_dictionary, utf8view_to_date32_dyn, utf8view_to_dictionary,
    utf8view_to_naive_timestamp_dyn, view_to_binary,
};
pub use binview_to::{binview_to_fixed_size_list_dyn, binview_to_primitive_dyn};
use dictionary_to::*;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_utils::IdxSize;
use polars_utils::float16::pf16;
pub use primitive_to::*;
use temporal::utf8view_to_timestamp;
pub use utf8_to::*;

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
        .map(|(arr, field)| cast(arr.as_ref(), field.dtype(), options))
        .collect::<PolarsResult<Vec<_>>>()?;

    Ok(StructArray::new(
        to_type.clone(),
        array.len(),
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

pub(super) fn cast_list_to_fixed_size_list<O: Offset>(
    list: &ListArray<O>,
    inner: &Field,
    size: usize,
    options: CastOptionsImpl,
) -> PolarsResult<FixedSizeListArray> {
    let null_cnt = list.null_count();
    let new_values = if null_cnt == 0 {
        let start_offset = list.offsets().first().to_usize();
        let offsets = list.offsets().buffer();

        let mut is_valid = true;
        for (i, offset) in offsets.iter().enumerate() {
            is_valid &= offset.to_usize() == start_offset + i * size;
        }

        polars_ensure!(is_valid, ComputeError: "not all elements have the specified width {size}");

        let sliced_values = list
            .values()
            .sliced(start_offset, list.offsets().range().to_usize());
        cast(sliced_values.as_ref(), inner.dtype(), options)?
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
        let take_values =
            unsafe { crate::gather::take_unchecked(list.values().as_ref(), &indices.freeze()) };

        cast(take_values.as_ref(), inner.dtype(), options)?
    };

    FixedSizeListArray::try_new(
        ArrowDataType::FixedSizeList(Box::new(inner.clone()), size),
        list.len(),
        new_values,
        list.validity().cloned(),
    )
    .map_err(|_| polars_err!(ComputeError: "not all elements have the specified width {size}"))
}

fn cast_list_uint8_to_binary<O: Offset>(list: &ListArray<O>) -> PolarsResult<BinaryViewArray> {
    let mut views = Vec::with_capacity(list.len());
    let mut result_validity = MutableBitmap::from_len_set(list.len());

    let u8array: &PrimitiveArray<u8> = list.values().as_any().downcast_ref().unwrap();
    let slice = u8array.values().as_slice();
    let mut cloned_buffers = vec![u8array.values().clone()];
    let mut buf_index = 0;
    let mut previous_buf_lengths = 0;
    let validity = list.validity();
    let internal_validity = list.values().validity();
    let offsets = list.offsets();

    let mut all_views_inline = true;

    // In a View for BinaryViewArray, both length and offset are u32.
    #[cfg(not(test))]
    const MAX_BUF_SIZE: usize = u32::MAX as usize;

    // This allows us to test some invariants without using 4GB of RAM; see mod
    // tests below.
    #[cfg(test)]
    const MAX_BUF_SIZE: usize = 15;

    for index in 0..list.len() {
        // Check if there's a null instead of a list:
        if let Some(validity) = validity {
            // SAFETY: We are generating indexes limited to < list.len().
            debug_assert!(index < validity.len());
            if unsafe { !validity.get_bit_unchecked(index) } {
                debug_assert!(index < result_validity.len());
                unsafe {
                    result_validity.set_unchecked(index, false);
                }
                views.push(View::default());
                continue;
            }
        }

        // SAFETY: We are generating indexes limited to < list.len().
        debug_assert!(index < offsets.len());
        let (start, end) = unsafe { offsets.start_end_unchecked(index) };
        let length = end - start;
        polars_ensure!(
            length <= MAX_BUF_SIZE,
            InvalidOperation:
            "when casting to BinaryView, list lengths must be <= {MAX_BUF_SIZE}"
        );

        // Check if the list contains nulls:
        if let Some(internal_validity) = internal_validity {
            if internal_validity.null_count_range(start, length) > 0 {
                debug_assert!(index < result_validity.len());
                unsafe {
                    result_validity.set_unchecked(index, false);
                }
                views.push(View::default());
                continue;
            }
        }

        if end - previous_buf_lengths > MAX_BUF_SIZE {
            // View offsets must fit in u32 (or smaller value when running Rust
            // tests), and we've determined the end of the next view will be
            // past that.
            buf_index += 1;
            let (previous, next) = cloned_buffers
                .last()
                .unwrap()
                .split_at(start - previous_buf_lengths);
            debug_assert!(previous.len() <= MAX_BUF_SIZE);
            previous_buf_lengths += previous.len();
            *(cloned_buffers.last_mut().unwrap()) = previous;
            cloned_buffers.push(next);
        }
        let view = View::new_from_bytes(
            &slice[start..end],
            buf_index,
            (start - previous_buf_lengths) as u32,
        );
        if !view.is_inline() {
            all_views_inline = false;
        }
        debug_assert_eq!(
            unsafe { view.get_slice_unchecked(&cloned_buffers) },
            &slice[start..end]
        );
        views.push(view);
    }

    // Optimization: don't actually need buffers if Views are all inline.
    if all_views_inline {
        cloned_buffers.clear();
    }

    let result = if cfg!(debug_assertions) {
        // A safer wrapper around new_unchecked_unknown_md; it shouldn't ever
        // fail in practice.
        BinaryViewArrayGeneric::try_new(
            ArrowDataType::BinaryView,
            views.into(),
            cloned_buffers.into(),
            result_validity.into(),
        )?
    } else {
        unsafe {
            BinaryViewArrayGeneric::new_unchecked_unknown_md(
                ArrowDataType::BinaryView,
                views.into(),
                cloned_buffers.into(),
                result_validity.into(),
                // We could compute this ourselves, but we want to make this code
                // match debug_assertions path as much as possible.
                None,
            )
        }
    };

    Ok(result)
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
/// * List of UInt8 to Binary: the list of integers becomes binary data, nulls in the list means it becomes a null
/// * Struct to Struct: the underlying fields are cast.
/// * PrimitiveArray to List: a list array with 1 value per slot is created
/// * Date32 and Date64: precision lost when going to higher interval
/// * Time32 and Time64: precision lost when going to higher interval
/// * Timestamp and Date{32|64}: precision lost when going to higher interval
/// * Temporal to/from backing primitive: zero-copy with data type change
///
/// Unsupported Casts
/// * non-`StructArray` to `StructArray` or `StructArray` to non-`StructArray`
/// * List to primitive (other than UInt8)
/// * Utf8 to boolean
/// * Interval and duration
pub fn cast(
    array: &dyn Array,
    to_type: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn Array>> {
    use ArrowDataType::*;
    let from_type = array.dtype();

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
        (List(field), BinaryView) if matches!(field.dtype(), UInt8) => {
            cast_list_uint8_to_binary::<i32>(array.as_any().downcast_ref().unwrap())
                .map(|arr| arr.boxed())
        },
        (LargeList(field), BinaryView) if matches!(field.dtype(), UInt8) => {
            cast_list_uint8_to_binary::<i64>(array.as_any().downcast_ref().unwrap())
                .map(|arr| arr.boxed())
        },
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
            LargeList(inner) if matches!(inner.dtype, ArrowDataType::UInt8) => {
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
            let values = cast(array, &to.dtype, options)?;
            // create offsets, where if array.len() = 2, we have [0,1,2]
            let offsets = (0..=array.len() as i32).collect::<Vec<_>>();
            // SAFETY: offsets _are_ monotonically increasing
            let offsets = unsafe { Offsets::new_unchecked(offsets) };

            let list_array = ListArray::<i32>::new(to_type.clone(), offsets.into(), values, None);

            Ok(Box::new(list_array))
        },

        (_, LargeList(to)) if from_type != &LargeBinary => {
            // cast primitive to list's primitive
            let values = cast(array, &to.dtype, options)?;
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
                UInt8 => utf8view_to_primitive_dyn::<u8>(arr, to_type, options),
                UInt16 => utf8view_to_primitive_dyn::<u16>(arr, to_type, options),
                UInt32 => utf8view_to_primitive_dyn::<u32>(arr, to_type, options),
                UInt64 => utf8view_to_primitive_dyn::<u64>(arr, to_type, options),
                #[cfg(feature = "dtype-u128")]
                UInt128 => utf8view_to_primitive_dyn::<u128>(arr, to_type, options),
                Int8 => utf8view_to_primitive_dyn::<i8>(arr, to_type, options),
                Int16 => utf8view_to_primitive_dyn::<i16>(arr, to_type, options),
                Int32 => utf8view_to_primitive_dyn::<i32>(arr, to_type, options),
                Int64 => utf8view_to_primitive_dyn::<i64>(arr, to_type, options),
                #[cfg(feature = "dtype-i128")]
                Int128 => utf8view_to_primitive_dyn::<i128>(arr, to_type, options),
                #[cfg(feature = "dtype-f16")]
                Float16 => utf8view_to_primitive_dyn::<pf16>(arr, to_type, options),
                Float32 => utf8view_to_primitive_dyn::<f32>(arr, to_type, options),
                Float64 => utf8view_to_primitive_dyn::<f64>(arr, to_type, options),
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
                #[cfg(feature = "dtype-decimal")]
                Decimal(precision, scale) => {
                    Ok(binview_to_decimal(&arr.to_binview(), *precision, *scale).to_boxed())
                },
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
            #[cfg(feature = "dtype-u128")]
            UInt128 => primitive_to_boolean_dyn::<u128>(array, to_type.clone()),
            Int8 => primitive_to_boolean_dyn::<i8>(array, to_type.clone()),
            Int16 => primitive_to_boolean_dyn::<i16>(array, to_type.clone()),
            Int32 => primitive_to_boolean_dyn::<i32>(array, to_type.clone()),
            Int64 => primitive_to_boolean_dyn::<i64>(array, to_type.clone()),
            #[cfg(feature = "dtype-i128")]
            Int128 => primitive_to_boolean_dyn::<i128>(array, to_type.clone()),
            #[cfg(feature = "dtype-f16")]
            Float16 => primitive_to_boolean_dyn::<pf16>(array, to_type.clone()),
            Float32 => primitive_to_boolean_dyn::<f32>(array, to_type.clone()),
            Float64 => primitive_to_boolean_dyn::<f64>(array, to_type.clone()),
            #[cfg(feature = "dtype-decimal")]
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
            #[cfg(feature = "dtype-u128")]
            UInt128 => boolean_to_primitive_dyn::<u128>(array),
            Int8 => boolean_to_primitive_dyn::<i8>(array),
            Int16 => boolean_to_primitive_dyn::<i16>(array),
            Int32 => boolean_to_primitive_dyn::<i32>(array),
            Int64 => boolean_to_primitive_dyn::<i64>(array),
            #[cfg(feature = "dtype-i128")]
            Int128 => boolean_to_primitive_dyn::<i128>(array),
            #[cfg(feature = "dtype-f16")]
            Float16 => boolean_to_primitive_dyn::<pf16>(array),
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
            #[cfg(feature = "dtype-u128")]
            UInt128 => binary_to_primitive_dyn::<i64, u128>(array, to_type, options),
            Int8 => binary_to_primitive_dyn::<i64, i8>(array, to_type, options),
            Int16 => binary_to_primitive_dyn::<i64, i16>(array, to_type, options),
            Int32 => binary_to_primitive_dyn::<i64, i32>(array, to_type, options),
            Int64 => binary_to_primitive_dyn::<i64, i64>(array, to_type, options),
            #[cfg(feature = "dtype-i128")]
            Int128 => binary_to_primitive_dyn::<i64, i128>(array, to_type, options),
            #[cfg(feature = "dtype-f16")]
            Float16 => binary_to_primitive_dyn::<i64, pf16>(array, to_type, options),
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
        #[cfg(feature = "dtype-u128")]
        (UInt8, UInt128) => primitive_to_primitive_dyn::<u8, u128>(array, to_type, options),
        (UInt8, Int8) => primitive_to_primitive_dyn::<u8, i8>(array, to_type, options),
        (UInt8, Int16) => primitive_to_primitive_dyn::<u8, i16>(array, to_type, options),
        (UInt8, Int32) => primitive_to_primitive_dyn::<u8, i32>(array, to_type, options),
        (UInt8, Int64) => primitive_to_primitive_dyn::<u8, i64>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (UInt8, Int128) => primitive_to_primitive_dyn::<u8, i128>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (UInt8, Float16) => primitive_to_primitive_dyn::<u8, pf16>(array, to_type, as_options),
        (UInt8, Float32) => primitive_to_primitive_dyn::<u8, f32>(array, to_type, as_options),
        (UInt8, Float64) => primitive_to_primitive_dyn::<u8, f64>(array, to_type, as_options),
        #[cfg(feature = "dtype-decimal")]
        (UInt8, Decimal(p, s)) => integer_to_decimal_dyn::<u8>(array, *p, *s),

        (UInt16, UInt8) => primitive_to_primitive_dyn::<u16, u8>(array, to_type, options),
        (UInt16, UInt32) => primitive_to_primitive_dyn::<u16, u32>(array, to_type, as_options),
        (UInt16, UInt64) => primitive_to_primitive_dyn::<u16, u64>(array, to_type, as_options),
        #[cfg(feature = "dtype-u128")]
        (UInt16, UInt128) => primitive_to_primitive_dyn::<u16, u128>(array, to_type, options),
        (UInt16, Int8) => primitive_to_primitive_dyn::<u16, i8>(array, to_type, options),
        (UInt16, Int16) => primitive_to_primitive_dyn::<u16, i16>(array, to_type, options),
        (UInt16, Int32) => primitive_to_primitive_dyn::<u16, i32>(array, to_type, options),
        (UInt16, Int64) => primitive_to_primitive_dyn::<u16, i64>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (UInt16, Int128) => primitive_to_primitive_dyn::<u16, i128>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (UInt16, Float16) => primitive_to_primitive_dyn::<u16, pf16>(array, to_type, as_options),
        (UInt16, Float32) => primitive_to_primitive_dyn::<u16, f32>(array, to_type, as_options),
        (UInt16, Float64) => primitive_to_primitive_dyn::<u16, f64>(array, to_type, as_options),
        #[cfg(feature = "dtype-decimal")]
        (UInt16, Decimal(p, s)) => integer_to_decimal_dyn::<u16>(array, *p, *s),

        (UInt32, UInt8) => primitive_to_primitive_dyn::<u32, u8>(array, to_type, options),
        (UInt32, UInt16) => primitive_to_primitive_dyn::<u32, u16>(array, to_type, options),
        (UInt32, UInt64) => primitive_to_primitive_dyn::<u32, u64>(array, to_type, as_options),
        #[cfg(feature = "dtype-u128")]
        (UInt32, UInt128) => primitive_to_primitive_dyn::<u32, u128>(array, to_type, options),
        (UInt32, Int8) => primitive_to_primitive_dyn::<u32, i8>(array, to_type, options),
        (UInt32, Int16) => primitive_to_primitive_dyn::<u32, i16>(array, to_type, options),
        (UInt32, Int32) => primitive_to_primitive_dyn::<u32, i32>(array, to_type, options),
        (UInt32, Int64) => primitive_to_primitive_dyn::<u32, i64>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (UInt32, Int128) => primitive_to_primitive_dyn::<u32, i128>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (UInt32, Float16) => primitive_to_primitive_dyn::<u32, pf16>(array, to_type, as_options),
        (UInt32, Float32) => primitive_to_primitive_dyn::<u32, f32>(array, to_type, as_options),
        (UInt32, Float64) => primitive_to_primitive_dyn::<u32, f64>(array, to_type, as_options),
        #[cfg(feature = "dtype-decimal")]
        (UInt32, Decimal(p, s)) => integer_to_decimal_dyn::<u32>(array, *p, *s),

        (UInt64, UInt8) => primitive_to_primitive_dyn::<u64, u8>(array, to_type, options),
        (UInt64, UInt16) => primitive_to_primitive_dyn::<u64, u16>(array, to_type, options),
        (UInt64, UInt32) => primitive_to_primitive_dyn::<u64, u32>(array, to_type, options),
        #[cfg(feature = "dtype-u128")]
        (UInt64, UInt128) => primitive_to_primitive_dyn::<u64, u128>(array, to_type, options),
        (UInt64, Int8) => primitive_to_primitive_dyn::<u64, i8>(array, to_type, options),
        (UInt64, Int16) => primitive_to_primitive_dyn::<u64, i16>(array, to_type, options),
        (UInt64, Int32) => primitive_to_primitive_dyn::<u64, i32>(array, to_type, options),
        (UInt64, Int64) => primitive_to_primitive_dyn::<u64, i64>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (UInt64, Int128) => primitive_to_primitive_dyn::<u64, i128>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (UInt64, Float16) => primitive_to_primitive_dyn::<u64, pf16>(array, to_type, as_options),
        (UInt64, Float32) => primitive_to_primitive_dyn::<u64, f32>(array, to_type, as_options),
        (UInt64, Float64) => primitive_to_primitive_dyn::<u64, f64>(array, to_type, as_options),
        #[cfg(feature = "dtype-decimal")]
        (UInt64, Decimal(p, s)) => integer_to_decimal_dyn::<u64>(array, *p, *s),

        #[cfg(feature = "dtype-u128")]
        (UInt128, UInt8) => primitive_to_primitive_dyn::<u128, u8>(array, to_type, options),
        #[cfg(feature = "dtype-u128")]
        (UInt128, UInt16) => primitive_to_primitive_dyn::<u128, u16>(array, to_type, options),
        #[cfg(feature = "dtype-u128")]
        (UInt128, UInt32) => primitive_to_primitive_dyn::<u128, u32>(array, to_type, options),
        #[cfg(feature = "dtype-u128")]
        (UInt128, UInt64) => primitive_to_primitive_dyn::<u128, u64>(array, to_type, options),
        #[cfg(feature = "dtype-u128")]
        (UInt128, Int8) => primitive_to_primitive_dyn::<u128, i8>(array, to_type, options),
        #[cfg(feature = "dtype-u128")]
        (UInt128, Int16) => primitive_to_primitive_dyn::<u128, i16>(array, to_type, options),
        #[cfg(feature = "dtype-u128")]
        (UInt128, Int32) => primitive_to_primitive_dyn::<u128, i32>(array, to_type, options),
        #[cfg(feature = "dtype-u128")]
        (UInt128, Int64) => primitive_to_primitive_dyn::<u128, i64>(array, to_type, options),
        #[cfg(all(feature = "dtype-u128", feature = "dtype-i128"))]
        (UInt128, Int128) => primitive_to_primitive_dyn::<u128, i128>(array, to_type, options),
        #[cfg(all(feature = "dtype-u128", feature = "dtype-f16"))]
        (UInt128, Float16) => primitive_to_primitive_dyn::<u128, pf16>(array, to_type, as_options),
        #[cfg(feature = "dtype-u128")]
        (UInt128, Float32) => primitive_to_primitive_dyn::<u128, f32>(array, to_type, as_options),
        #[cfg(feature = "dtype-u128")]
        (UInt128, Float64) => primitive_to_primitive_dyn::<u128, f64>(array, to_type, as_options),
        #[cfg(all(feature = "dtype-u128", feature = "dtype-decimal"))]
        (UInt128, Decimal(p, s)) => integer_to_decimal_dyn::<u128>(array, *p, *s),

        (Int8, UInt8) => primitive_to_primitive_dyn::<i8, u8>(array, to_type, options),
        (Int8, UInt16) => primitive_to_primitive_dyn::<i8, u16>(array, to_type, options),
        (Int8, UInt32) => primitive_to_primitive_dyn::<i8, u32>(array, to_type, options),
        (Int8, UInt64) => primitive_to_primitive_dyn::<i8, u64>(array, to_type, options),
        #[cfg(feature = "dtype-u128")]
        (Int8, UInt128) => primitive_to_primitive_dyn::<i8, u128>(array, to_type, options),
        (Int8, Int16) => primitive_to_primitive_dyn::<i8, i16>(array, to_type, as_options),
        (Int8, Int32) => primitive_to_primitive_dyn::<i8, i32>(array, to_type, as_options),
        (Int8, Int64) => primitive_to_primitive_dyn::<i8, i64>(array, to_type, as_options),
        #[cfg(feature = "dtype-i128")]
        (Int8, Int128) => primitive_to_primitive_dyn::<i8, i128>(array, to_type, as_options),
        #[cfg(feature = "dtype-f16")]
        (Int8, Float16) => primitive_to_primitive_dyn::<i8, pf16>(array, to_type, as_options),
        (Int8, Float32) => primitive_to_primitive_dyn::<i8, f32>(array, to_type, as_options),
        (Int8, Float64) => primitive_to_primitive_dyn::<i8, f64>(array, to_type, as_options),
        #[cfg(feature = "dtype-decimal")]
        (Int8, Decimal(p, s)) => integer_to_decimal_dyn::<i8>(array, *p, *s),

        (Int16, UInt8) => primitive_to_primitive_dyn::<i16, u8>(array, to_type, options),
        (Int16, UInt16) => primitive_to_primitive_dyn::<i16, u16>(array, to_type, options),
        (Int16, UInt32) => primitive_to_primitive_dyn::<i16, u32>(array, to_type, options),
        (Int16, UInt64) => primitive_to_primitive_dyn::<i16, u64>(array, to_type, options),
        #[cfg(feature = "dtype-u128")]
        (Int16, UInt128) => primitive_to_primitive_dyn::<i16, u128>(array, to_type, options),
        (Int16, Int8) => primitive_to_primitive_dyn::<i16, i8>(array, to_type, options),
        (Int16, Int32) => primitive_to_primitive_dyn::<i16, i32>(array, to_type, as_options),
        (Int16, Int64) => primitive_to_primitive_dyn::<i16, i64>(array, to_type, as_options),
        #[cfg(feature = "dtype-i128")]
        (Int16, Int128) => primitive_to_primitive_dyn::<i16, i128>(array, to_type, as_options),
        #[cfg(feature = "dtype-f16")]
        (Int16, Float16) => primitive_to_primitive_dyn::<i16, pf16>(array, to_type, as_options),
        (Int16, Float32) => primitive_to_primitive_dyn::<i16, f32>(array, to_type, as_options),
        (Int16, Float64) => primitive_to_primitive_dyn::<i16, f64>(array, to_type, as_options),
        #[cfg(feature = "dtype-decimal")]
        (Int16, Decimal(p, s)) => integer_to_decimal_dyn::<i16>(array, *p, *s),

        (Int32, UInt8) => primitive_to_primitive_dyn::<i32, u8>(array, to_type, options),
        (Int32, UInt16) => primitive_to_primitive_dyn::<i32, u16>(array, to_type, options),
        (Int32, UInt32) => primitive_to_primitive_dyn::<i32, u32>(array, to_type, options),
        (Int32, UInt64) => primitive_to_primitive_dyn::<i32, u64>(array, to_type, options),
        #[cfg(feature = "dtype-u128")]
        (Int32, UInt128) => primitive_to_primitive_dyn::<i32, u128>(array, to_type, options),
        (Int32, Int8) => primitive_to_primitive_dyn::<i32, i8>(array, to_type, options),
        (Int32, Int16) => primitive_to_primitive_dyn::<i32, i16>(array, to_type, options),
        (Int32, Int64) => primitive_to_primitive_dyn::<i32, i64>(array, to_type, as_options),
        #[cfg(feature = "dtype-i128")]
        (Int32, Int128) => primitive_to_primitive_dyn::<i32, i128>(array, to_type, as_options),
        #[cfg(feature = "dtype-f16")]
        (Int32, Float16) => primitive_to_primitive_dyn::<i32, pf16>(array, to_type, as_options),
        (Int32, Float32) => primitive_to_primitive_dyn::<i32, f32>(array, to_type, as_options),
        (Int32, Float64) => primitive_to_primitive_dyn::<i32, f64>(array, to_type, as_options),
        #[cfg(feature = "dtype-decimal")]
        (Int32, Decimal(p, s)) => integer_to_decimal_dyn::<i32>(array, *p, *s),

        (Int64, UInt8) => primitive_to_primitive_dyn::<i64, u8>(array, to_type, options),
        (Int64, UInt16) => primitive_to_primitive_dyn::<i64, u16>(array, to_type, options),
        (Int64, UInt32) => primitive_to_primitive_dyn::<i64, u32>(array, to_type, options),
        (Int64, UInt64) => primitive_to_primitive_dyn::<i64, u64>(array, to_type, options),
        #[cfg(feature = "dtype-u128")]
        (Int64, UInt128) => primitive_to_primitive_dyn::<i64, u128>(array, to_type, options),
        (Int64, Int8) => primitive_to_primitive_dyn::<i64, i8>(array, to_type, options),
        (Int64, Int16) => primitive_to_primitive_dyn::<i64, i16>(array, to_type, options),
        (Int64, Int32) => primitive_to_primitive_dyn::<i64, i32>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (Int64, Int128) => primitive_to_primitive_dyn::<i64, i128>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (Int64, Float16) => primitive_to_primitive_dyn::<i64, pf16>(array, to_type, as_options),
        (Int64, Float32) => primitive_to_primitive_dyn::<i64, f32>(array, to_type, options),
        (Int64, Float64) => primitive_to_primitive_dyn::<i64, f64>(array, to_type, as_options),
        #[cfg(feature = "dtype-decimal")]
        (Int64, Decimal(p, s)) => integer_to_decimal_dyn::<i64>(array, *p, *s),

        #[cfg(feature = "dtype-i128")]
        (Int128, UInt8) => primitive_to_primitive_dyn::<i128, u8>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (Int128, UInt16) => primitive_to_primitive_dyn::<i128, u16>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (Int128, UInt32) => primitive_to_primitive_dyn::<i128, u32>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (Int128, UInt64) => primitive_to_primitive_dyn::<i128, u64>(array, to_type, options),
        #[cfg(all(feature = "dtype-u128", feature = "dtype-i128"))]
        (Int128, UInt128) => primitive_to_primitive_dyn::<i128, u128>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (Int128, Int8) => primitive_to_primitive_dyn::<i128, i8>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (Int128, Int16) => primitive_to_primitive_dyn::<i128, i16>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (Int128, Int32) => primitive_to_primitive_dyn::<i128, i32>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (Int128, Int64) => primitive_to_primitive_dyn::<i128, i64>(array, to_type, options),
        #[cfg(all(feature = "dtype-i128", feature = "dtype-f16"))]
        (Int128, Float16) => primitive_to_primitive_dyn::<i128, pf16>(array, to_type, as_options),
        #[cfg(feature = "dtype-i128")]
        (Int128, Float32) => primitive_to_primitive_dyn::<i128, f32>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (Int128, Float64) => primitive_to_primitive_dyn::<i128, f64>(array, to_type, as_options),
        #[cfg(all(feature = "dtype-i128", feature = "dtype-decimal"))]
        (Int128, Decimal(p, s)) => integer_to_decimal_dyn::<i128>(array, *p, *s),

        #[cfg(feature = "dtype-f16")]
        (Float16, UInt8) => primitive_to_primitive_dyn::<pf16, u8>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (Float16, UInt16) => primitive_to_primitive_dyn::<pf16, u16>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (Float16, UInt32) => primitive_to_primitive_dyn::<pf16, u32>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (Float16, UInt64) => primitive_to_primitive_dyn::<pf16, u64>(array, to_type, options),
        #[cfg(all(feature = "dtype-f16", feature = "dtype-u128"))]
        (Float16, UInt128) => primitive_to_primitive_dyn::<pf16, u128>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (Float16, Int8) => primitive_to_primitive_dyn::<pf16, i8>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (Float16, Int16) => primitive_to_primitive_dyn::<pf16, i16>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (Float16, Int32) => primitive_to_primitive_dyn::<pf16, i32>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (Float16, Int64) => primitive_to_primitive_dyn::<pf16, i64>(array, to_type, options),
        #[cfg(all(feature = "dtype-f16", feature = "dtype-i128"))]
        (Float16, Int128) => primitive_to_primitive_dyn::<pf16, i128>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (Float16, Float32) => primitive_to_primitive_dyn::<pf16, f32>(array, to_type, as_options),
        #[cfg(feature = "dtype-f16")]
        (Float16, Float64) => primitive_to_primitive_dyn::<pf16, f64>(array, to_type, as_options),
        #[cfg(all(feature = "dtype-f16", feature = "dtype-decimal"))]
        (Float16, Decimal(p, s)) => float_to_decimal_dyn::<pf16>(array, *p, *s),

        (Float32, UInt8) => primitive_to_primitive_dyn::<f32, u8>(array, to_type, options),
        (Float32, UInt16) => primitive_to_primitive_dyn::<f32, u16>(array, to_type, options),
        (Float32, UInt32) => primitive_to_primitive_dyn::<f32, u32>(array, to_type, options),
        (Float32, UInt64) => primitive_to_primitive_dyn::<f32, u64>(array, to_type, options),
        #[cfg(feature = "dtype-u128")]
        (Float32, UInt128) => primitive_to_primitive_dyn::<f32, u128>(array, to_type, options),
        (Float32, Int8) => primitive_to_primitive_dyn::<f32, i8>(array, to_type, options),
        (Float32, Int16) => primitive_to_primitive_dyn::<f32, i16>(array, to_type, options),
        (Float32, Int32) => primitive_to_primitive_dyn::<f32, i32>(array, to_type, options),
        (Float32, Int64) => primitive_to_primitive_dyn::<f32, i64>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (Float32, Int128) => primitive_to_primitive_dyn::<f32, i128>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (Float32, Float16) => primitive_to_primitive_dyn::<f32, pf16>(array, to_type, as_options),
        (Float32, Float64) => primitive_to_primitive_dyn::<f32, f64>(array, to_type, as_options),
        #[cfg(feature = "dtype-decimal")]
        (Float32, Decimal(p, s)) => float_to_decimal_dyn::<f32>(array, *p, *s),

        (Float64, UInt8) => primitive_to_primitive_dyn::<f64, u8>(array, to_type, options),
        (Float64, UInt16) => primitive_to_primitive_dyn::<f64, u16>(array, to_type, options),
        (Float64, UInt32) => primitive_to_primitive_dyn::<f64, u32>(array, to_type, options),
        (Float64, UInt64) => primitive_to_primitive_dyn::<f64, u64>(array, to_type, options),
        #[cfg(feature = "dtype-u128")]
        (Float64, UInt128) => primitive_to_primitive_dyn::<f64, u128>(array, to_type, options),
        (Float64, Int8) => primitive_to_primitive_dyn::<f64, i8>(array, to_type, options),
        (Float64, Int16) => primitive_to_primitive_dyn::<f64, i16>(array, to_type, options),
        (Float64, Int32) => primitive_to_primitive_dyn::<f64, i32>(array, to_type, options),
        (Float64, Int64) => primitive_to_primitive_dyn::<f64, i64>(array, to_type, options),
        #[cfg(feature = "dtype-i128")]
        (Float64, Int128) => primitive_to_primitive_dyn::<f64, i128>(array, to_type, options),
        #[cfg(feature = "dtype-f16")]
        (Float64, Float16) => primitive_to_primitive_dyn::<f64, pf16>(array, to_type, as_options),
        (Float64, Float32) => primitive_to_primitive_dyn::<f64, f32>(array, to_type, options),
        #[cfg(feature = "dtype-decimal")]
        (Float64, Decimal(p, s)) => float_to_decimal_dyn::<f64>(array, *p, *s),

        #[cfg(feature = "dtype-decimal")]
        (Decimal(_, _), UInt8) => decimal_to_integer_dyn::<u8>(array),
        #[cfg(feature = "dtype-decimal")]
        (Decimal(_, _), UInt16) => decimal_to_integer_dyn::<u16>(array),
        #[cfg(feature = "dtype-decimal")]
        (Decimal(_, _), UInt32) => decimal_to_integer_dyn::<u32>(array),
        #[cfg(feature = "dtype-decimal")]
        (Decimal(_, _), UInt64) => decimal_to_integer_dyn::<u64>(array),
        #[cfg(all(feature = "dtype-decimal", feature = "dtype-u128"))]
        (Decimal(_, _), UInt128) => decimal_to_integer_dyn::<u128>(array),
        #[cfg(feature = "dtype-decimal")]
        (Decimal(_, _), Int8) => decimal_to_integer_dyn::<i8>(array),
        #[cfg(feature = "dtype-decimal")]
        (Decimal(_, _), Int16) => decimal_to_integer_dyn::<i16>(array),
        #[cfg(feature = "dtype-decimal")]
        (Decimal(_, _), Int32) => decimal_to_integer_dyn::<i32>(array),
        #[cfg(feature = "dtype-decimal")]
        (Decimal(_, _), Int64) => decimal_to_integer_dyn::<i64>(array),
        #[cfg(all(feature = "dtype-decimal", feature = "dtype-i128"))]
        (Decimal(_, _), Int128) => decimal_to_integer_dyn::<i128>(array),
        #[cfg(all(feature = "dtype-decimal", feature = "dtype-f16"))]
        (Decimal(_, _), Float16) => decimal_to_float_dyn::<pf16>(array),
        #[cfg(feature = "dtype-decimal")]
        (Decimal(_, _), Float32) => decimal_to_float_dyn::<f32>(array),
        #[cfg(feature = "dtype-decimal")]
        (Decimal(_, _), Float64) => decimal_to_float_dyn::<f64>(array),
        #[cfg(feature = "dtype-decimal")]
        (Decimal(_, _), Decimal(to_p, to_s)) => decimal_to_decimal_dyn(array, *to_p, *to_s),
        // end numeric casts

        // temporal casts
        (Int32, Date32) => primitive_to_same_primitive_dyn::<i32>(array, to_type),
        (Int32, Time32(TimeUnit::Second)) => primitive_dyn!(array, int32_to_time32s),
        (Int32, Time32(TimeUnit::Millisecond)) => primitive_dyn!(array, int32_to_time32ms),
        // No support for microsecond/nanosecond with i32
        (Date32, Int32) => primitive_to_same_primitive_dyn::<i32>(array, to_type),
        (Date32, Int64) => primitive_to_primitive_dyn::<i32, i64>(array, to_type, options),
        (Time32(_), Int32) => primitive_to_same_primitive_dyn::<i32>(array, to_type),
        (Int64, Date64) => primitive_to_same_primitive_dyn::<i64>(array, to_type),
        // No support for second/milliseconds with i64
        (Int64, Time64(TimeUnit::Microsecond)) => primitive_dyn!(array, int64_to_time64us),
        (Int64, Time64(TimeUnit::Nanosecond)) => primitive_dyn!(array, int64_to_time64ns),

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
    match dict_value_type.to_storage() {
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
        UInt128 => primitive_to_binview_dyn::<u128>(array),
        Int8 => primitive_to_binview_dyn::<i8>(array),
        Int16 => primitive_to_binview_dyn::<i16>(array),
        Int32 => primitive_to_binview_dyn::<i32>(array),
        Int64 => primitive_to_binview_dyn::<i64>(array),
        Int128 => primitive_to_binview_dyn::<i128>(array),
        Float16 => primitive_to_binview_dyn::<pf16>(array),
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

#[cfg(test)]
mod tests {
    use arrow::offset::OffsetsBuffer;
    use polars_error::PolarsError;

    use super::*;

    /// When cfg(test), offsets for ``View``s generated by
    /// cast_list_uint8_to_binary() are limited to max value of 3, so buffers
    /// need to be split aggressively.
    #[test]
    fn cast_list_uint8_to_binary_across_buffer_max_size() {
        let dtype =
            ArrowDataType::List(Box::new(Field::new("".into(), ArrowDataType::UInt8, true)));
        let values = PrimitiveArray::from_slice((0u8..20).collect::<Vec<_>>()).boxed();
        let list_u8 = ListArray::try_new(
            dtype,
            unsafe { OffsetsBuffer::new_unchecked(vec![0, 13, 18, 20].into()) },
            values,
            None,
        )
        .unwrap();

        let binary = cast(
            &list_u8,
            &ArrowDataType::BinaryView,
            CastOptionsImpl::default(),
        )
        .unwrap();
        let binary_array: &BinaryViewArray = binary.as_ref().as_any().downcast_ref().unwrap();
        assert_eq!(
            binary_array
                .values_iter()
                .map(|s| s.to_vec())
                .collect::<Vec<Vec<u8>>>(),
            vec![
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                vec![13, 14, 15, 16, 17],
                vec![18, 19]
            ]
        );
        // max offset of 15 so we need to split:
        assert_eq!(
            binary_array
                .data_buffers()
                .iter()
                .map(|buf| buf.len())
                .collect::<Vec<_>>(),
            vec![13, 7]
        );
    }

    /// Arrow spec requires views to fit in a single buffer. When cfg(test),
    /// buffers generated by cast_list_uint8_to_binary are of size 15 or
    /// smaller, so a list of size 16 should cause an error.
    #[test]
    fn cast_list_uint8_to_binary_errors_too_large_list() {
        let values = PrimitiveArray::from_slice(vec![0u8; 16]);
        let dtype =
            ArrowDataType::List(Box::new(Field::new("".into(), ArrowDataType::UInt8, true)));
        let list_u8 = ListArray::new(
            dtype,
            OffsetsBuffer::one_with_length(16),
            values.boxed(),
            None,
        );

        let err = cast(
            &list_u8,
            &ArrowDataType::BinaryView,
            CastOptionsImpl::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            PolarsError::InvalidOperation(msg)
                if msg.as_ref() == "when casting to BinaryView, list lengths must be <= 15"
        ));
    }

    /// When all views are <=12, cast_list_uint8_to_binary drops buffers in the
    /// result because all views are inline.
    #[test]
    fn cast_list_uint8_to_binary_drops_small_buffers() {
        let values = PrimitiveArray::from_slice(vec![10u8; 12]);
        let dtype =
            ArrowDataType::List(Box::new(Field::new("".into(), ArrowDataType::UInt8, true)));
        let list_u8 = ListArray::new(
            dtype,
            OffsetsBuffer::one_with_length(12),
            values.boxed(),
            None,
        );
        let binary = cast(
            &list_u8,
            &ArrowDataType::BinaryView,
            CastOptionsImpl::default(),
        )
        .unwrap();
        let binary_array: &BinaryViewArray = binary.as_ref().as_any().downcast_ref().unwrap();
        assert!(binary_array.data_buffers().is_empty());
        assert_eq!(
            binary_array
                .values_iter()
                .map(|s| s.to_vec())
                .collect::<Vec<Vec<u8>>>(),
            vec![vec![10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],]
        );
    }
}
