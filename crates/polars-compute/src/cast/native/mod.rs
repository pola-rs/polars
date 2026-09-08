//! Casting the arrays of `polars-array` on a pair of [`DataType`]s, without crossing over to Arrow.

mod binary;
mod binview;
mod boolean;
mod nested;
mod primitive;

use arrow::with_match_primitive_type;
use polars_array::bitmap::combine_validities_and;
use polars_array::{
    PlArray, PlArrayType, PlBinaryArray, PlBinaryViewArray, PlBitmap, PlBitmapRef, PlBooleanArray,
    PlNullArray, PlPrimitiveArray, PlUtf8ViewArray,
};
use polars_dtype::DataType;
use polars_error::{PolarsResult, polars_bail, polars_ensure};
pub use primitive::SerPrimitive;

pub use self::binary::{binary_to_binview, binary_to_list, fixed_size_binary_to_binview};
#[cfg(feature = "dtype-decimal")]
pub use self::binview::binview_to_decimal;
pub use self::binview::{
    binview_to_fixed_binary, binview_to_fixed_size_list, binview_to_primitive, view_to_binary,
};
#[cfg(feature = "dtype-struct")]
pub use self::nested::cast_struct;
#[cfg(feature = "dtype-array")]
pub use self::nested::{cast_fixed_size_list, fixed_size_list_to_list};
pub use self::nested::{cast_list, list_to_fixed_size_list, list_uint8_to_binview};
#[cfg(feature = "dtype-decimal")]
pub use self::primitive::decimal_to_utf8view;
pub use self::primitive::{
    boolean_to_primitive, numeric_to_numeric, numeric_to_numeric_checked, primitive_to_binview,
    primitive_to_boolean, rescale_time,
};
use super::CastOptionsImpl;

macro_rules! with_match_numeric_dtype {(
    $dtype:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    #[allow(unused_imports)]
    use polars_utils::float16::pf16;
    use polars_dtype::DataType::*;

    match $dtype {
        UInt8 => __with_ty__! { u8 },
        UInt16 => __with_ty__! { u16 },
        UInt32 => __with_ty__! { u32 },
        UInt64 => __with_ty__! { u64 },
        #[cfg(feature = "dtype-u128")]
        UInt128 => __with_ty__! { u128 },
        Int8 => __with_ty__! { i8 },
        Int16 => __with_ty__! { i16 },
        Int32 => __with_ty__! { i32 },
        Int64 => __with_ty__! { i64 },
        #[cfg(feature = "dtype-i128")]
        Int128 => __with_ty__! { i128 },
        #[cfg(feature = "dtype-f16")]
        Float16 => __with_ty__! { pf16 },
        Float32 => __with_ty__! { f32 },
        Float64 => __with_ty__! { f64 },
        dtype => unreachable!("a plain numeric type is one of the above, got {dtype:?}"),
    }
})}

#[cfg_attr(not(feature = "dtype-decimal"), allow(unused_macros))]
macro_rules! with_match_integer_dtype {(
    $dtype:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use polars_dtype::DataType::*;

    match $dtype {
        UInt8 => __with_ty__! { u8 },
        UInt16 => __with_ty__! { u16 },
        UInt32 => __with_ty__! { u32 },
        UInt64 => __with_ty__! { u64 },
        #[cfg(feature = "dtype-u128")]
        UInt128 => __with_ty__! { u128 },
        Int8 => __with_ty__! { i8 },
        Int16 => __with_ty__! { i16 },
        Int32 => __with_ty__! { i32 },
        Int64 => __with_ty__! { i64 },
        #[cfg(feature = "dtype-i128")]
        Int128 => __with_ty__! { i128 },
        dtype => unreachable!("an integer type is one of the above, got {dtype:?}"),
    }
})}

#[cfg_attr(not(feature = "dtype-decimal"), allow(unused_macros))]
macro_rules! with_match_float_dtype {(
    $dtype:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    #[allow(unused_imports)]
    use polars_utils::float16::pf16;
    use polars_dtype::DataType::*;

    match $dtype {
        #[cfg(feature = "dtype-f16")]
        Float16 => __with_ty__! { pf16 },
        Float32 => __with_ty__! { f32 },
        Float64 => __with_ty__! { f64 },
        dtype => unreachable!("a float type is one of the above, got {dtype:?}"),
    }
})}

/// Casts `array`, which holds the elements of `from`, to `to`.
pub fn cast(
    array: &dyn PlArray,
    from: &DataType,
    to: &DataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn PlArray>> {
    // A cast that changes nothing but the name over the values reads the same values, and an array
    // holds no name to change: the array *is* the answer.
    if from == to || is_retag(from, to) {
        return Ok(array.to_boxed());
    }

    // Every element of a scalar chunk is the same one, so what a cast answers for one of them is
    // what it answers for all of them — a cast reads one element at a time.
    if array.is_scalar() && array.len() > 1 {
        let element = cast(&*array.sliced(0, 1), from, to, options)?;
        return Ok(element.new_from_index(0, array.len()));
    }

    // A type that is nothing but a name over its physical type holds the values of that type, so a
    // cast off it reads them — unless the target is such a name too, where the pair is between the
    // names rather than the values.
    if wraps_its_physical_type(from) && !wraps_its_physical_type(to) {
        return cast(array, &from.to_physical(), to, options);
    }

    cast_dispatch(array, from, to, options)
}

fn cast_dispatch(
    array: &dyn PlArray,
    from: &DataType,
    to: &DataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn PlArray>> {
    use DataType as D;

    // Null on either side reads as null everywhere, which needs no slot per element.
    if matches!(from, D::Null) {
        return full_null(to, array.len());
    }
    if matches!(to, D::Null) {
        return Ok(Box::new(PlNullArray::new(array.len())));
    }

    #[cfg(feature = "dtype-struct")]
    if let (D::Struct(from_fields), D::Struct(to_fields)) = (from, to) {
        polars_ensure!(
            from_fields.len() == to_fields.len(),
            InvalidOperation: "Cannot cast struct with different number of fields."
        );
        return nested::cast_struct(downcast(array), |i, field| {
            cast(field, from_fields[i].dtype(), to_fields[i].dtype(), options)
        })
        .map(|array| Box::new(array) as _);
    }
    if from.is_struct() || to.is_struct() {
        polars_bail!(InvalidOperation: "Cannot cast from struct to other types");
    }

    if let Some(out) = cast_nested(array, from, to, options)? {
        return Ok(out);
    }

    if matches!(from, D::Boolean) {
        let array: &PlBooleanArray = downcast(array);
        if is_plain_numeric(to) {
            return Ok(with_match_numeric_dtype!(to, |$T| {
                Box::new(primitive::boolean_to_primitive::<$T>(array)) as Box<dyn PlArray>
            }));
        }
        return match to {
            D::String => Ok(Box::new(boolean::boolean_to_utf8view(array))),
            D::Binary => Ok(Box::new(boolean::boolean_to_binaryview(array))),
            _ => unsupported(from, to),
        };
    }

    if matches!(to, D::Boolean) {
        if is_plain_numeric(from) {
            return Ok(with_match_numeric_dtype!(from, |$T| {
                Box::new(primitive::primitive_to_boolean::<$T>(downcast(array))) as Box<dyn PlArray>
            }));
        }
        #[cfg(feature = "dtype-decimal")]
        if matches!(from, D::Decimal(_, _)) {
            return Ok(Box::new(primitive::primitive_to_boolean::<i128>(downcast(
                array,
            ))));
        }
        return unsupported(from, to);
    }

    if matches!(from, D::String) {
        return cast_string(downcast::<PlUtf8ViewArray>(array), to, options);
    }

    if matches!(from, D::Binary | D::BinaryOffset) {
        return cast_bytes(array, from, to, options);
    }

    if is_plain_numeric(from) {
        return cast_number(array, from, to, options);
    }

    #[cfg(feature = "dtype-decimal")]
    if let D::Decimal(_, from_scale) = from {
        let array: &PlPrimitiveArray<i128> = downcast(array);
        let from_scale = *from_scale;

        if to.is_float() {
            return Ok(with_match_float_dtype!(to, |$T| {
                Box::new(primitive::decimal_to_float::<$T>(array, from_scale)) as Box<dyn PlArray>
            }));
        }
        if to.is_integer() {
            return Ok(with_match_integer_dtype!(to, |$T| {
                Box::new(primitive::decimal_to_integer::<$T>(array, from_scale)) as Box<dyn PlArray>
            }));
        }
        return match to {
            D::Decimal(to_precision, to_scale) => {
                let D::Decimal(from_precision, _) = from else {
                    unreachable!("the decimal was matched on")
                };
                Ok(Box::new(primitive::decimal_to_decimal(
                    array,
                    *from_precision,
                    from_scale,
                    *to_precision,
                    *to_scale,
                )))
            },
            D::String => Ok(Box::new(primitive::decimal_to_utf8view(array, from_scale))),
            _ => unsupported(from, to),
        };
    }

    // What is left is a pair of names over the same physical type, which the values are read for.
    cast_temporal(array, from, to)
}

/// Casts between the temporal types, whose values stand for an elapsed time in some unit.
fn cast_temporal(
    array: &dyn PlArray,
    from: &DataType,
    to: &DataType,
) -> PolarsResult<Box<dyn PlArray>> {
    use DataType as D;

    match (from, to) {
        // Not a conversion but a range check: a time holds a day's worth of nanoseconds, so an
        // `i64` outside that range names no time and reads as null.
        (D::Int64, D::Time) => Ok(Box::new(primitive::int64_to_time(downcast(array)))),

        (D::Datetime(from_unit, _), D::Datetime(to_unit, _))
        | (D::Duration(from_unit), D::Duration(to_unit)) => Ok(Box::new(primitive::rescale_time(
            downcast(array),
            time_unit_multiple(*from_unit),
            time_unit_multiple(*to_unit),
        ))),
        (D::Datetime(from_unit, _), D::Date) => Ok(Box::new(primitive::timestamp_to_date(
            downcast(array),
            time_unit_multiple(*from_unit) * SECONDS_IN_DAY,
        ))),
        _ => unsupported(from, to),
    }
}

/// Casts a string array, whose elements are the text of what they hold.
fn cast_string(
    array: &PlUtf8ViewArray,
    to: &DataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn PlArray>> {
    use DataType as D;

    if is_plain_numeric(to) {
        return Ok(with_match_numeric_dtype!(to, |$T| {
            Box::new(binview::binview_to_parsed::<$T>(&array.clone().into_binview(), options))
                as Box<dyn PlArray>
        }));
    }

    match to {
        D::Binary => Ok(Box::new(array.clone().into_binview())),
        D::BinaryOffset => Ok(Box::new(binview::view_to_binary(
            &array.clone().into_binview(),
        ))),
        #[cfg(feature = "dtype-decimal")]
        D::Decimal(precision, scale) => Ok(Box::new(binview::binview_to_decimal(
            &array.clone().into_binview(),
            *precision,
            *scale,
        ))),
        D::Datetime(_, None) => polars_bail!(
            InvalidOperation:
            "casting from string to datetime is not supported.\n\
            It was removed in Polars 2.0. Use `str.to_datetime()` instead."
        ),
        D::Datetime(_, Some(time_zone)) => polars_bail!(
            InvalidOperation:
            "casting from string to datetime is not supported.\n\
            It was removed in Polars 2.0. Use `str.to_datetime(..., \"time_zone={time_zone}\")` instead."
        ),
        D::Date => polars_bail!(
            InvalidOperation:
            "casting from string to date is not supported.\n\
            It was removed in Polars 2.0. Use `str.to_date()` instead."
        ),
        D::Time => polars_bail!(
            InvalidOperation:
            "casting from string to time is not supported.\n\
            It was removed in Polars 2.0. Use `str.to_time()` instead."
        ),
        _ => unsupported(&DataType::String, to),
    }
}

/// Casts a binary array, whose elements are the bytes of what they hold.
fn cast_bytes(
    array: &dyn PlArray,
    from: &DataType,
    to: &DataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn PlArray>> {
    use DataType as D;

    if is_plain_numeric(to) {
        // The bytes of an offset-backed binary are read as the text of a number; those of a
        // `Binary` are not, which the Arrow kernels this replaced settled the same way.
        polars_ensure!(
            matches!(from, D::BinaryOffset),
            InvalidOperation: "casting from {from:?} to {to:?} not supported"
        );
        return Ok(with_match_numeric_dtype!(to, |$T| {
            Box::new(binary::binary_to_parsed::<$T>(downcast(array), options)) as Box<dyn PlArray>
        }));
    }

    // The bytes of both are read the same way, so the views are what every cast left reads.
    let converted;
    let view: &PlBinaryViewArray = match from {
        D::Binary => downcast(array),
        _ => {
            converted = binary::binary_to_binview(downcast::<PlBinaryArray>(array));
            &converted
        },
    };

    match to {
        // SAFETY: the caller of a cast to a string promises the bytes are valid UTF-8, which is
        // the invariant the Arrow kernels this replaced upheld the same way.
        D::String => Ok(Box::new(unsafe {
            PlUtf8ViewArray::from_binview_unchecked(view.clone())
        })),
        D::Binary => Ok(Box::new(view.clone())),
        D::BinaryOffset => Ok(Box::new(binview::view_to_binary(view))),
        _ => unsupported(from, to),
    }
}

/// Casts a number to another type, whose values it is read as.
fn cast_number(
    array: &dyn PlArray,
    from: &DataType,
    to: &DataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn PlArray>> {
    use DataType as D;

    if is_plain_numeric(to) {
        return Ok(with_match_numeric_dtype!(from, |$I| {
            let array: &PlPrimitiveArray<$I> = downcast(array);
            with_match_numeric_dtype!(to, |$O| {
                let wrapped = options.wrapped || casts_with_as(from, to);
                Box::new(primitive::numeric_to_numeric::<$I, $O>(array, wrapped)) as Box<dyn PlArray>
            })
        }));
    }

    match to {
        D::String => Ok(with_match_numeric_dtype!(from, |$T| {
            Box::new(primitive::primitive_to_utf8view::<$T>(downcast(array))) as Box<dyn PlArray>
        })),
        D::Binary => Ok(with_match_numeric_dtype!(from, |$T| {
            Box::new(primitive::primitive_to_binview::<$T>(downcast(array))) as Box<dyn PlArray>
        })),
        #[cfg(feature = "dtype-decimal")]
        D::Decimal(precision, scale) => {
            let (precision, scale) = (*precision, *scale);
            if from.is_float() {
                return Ok(with_match_float_dtype!(from, |$T| {
                    Box::new(primitive::float_to_decimal::<$T>(downcast(array), precision, scale))
                        as Box<dyn PlArray>
                }));
            }
            Ok(with_match_integer_dtype!(from, |$T| {
                Box::new(primitive::integer_to_decimal::<$T>(downcast(array), precision, scale))
                    as Box<dyn PlArray>
            }))
        },
        _ => cast_temporal(array, from, to),
    }
}

/// Casts the nested types, which hold the values of their children, or `None` for another pair.
fn cast_nested(
    array: &dyn PlArray,
    from: &DataType,
    to: &DataType,
    options: CastOptionsImpl,
) -> PolarsResult<Option<Box<dyn PlArray>>> {
    use DataType as D;

    let out: Box<dyn PlArray> = match (from, to) {
        (D::List(from_inner), D::List(to_inner)) => {
            Box::new(nested::cast_list(downcast(array), |values| {
                cast(values, from_inner, to_inner, options)
            })?)
        },
        #[cfg(feature = "dtype-array")]
        (D::Array(from_inner, from_width), D::Array(to_inner, to_width)) => {
            polars_ensure!(
                from_width == to_width,
                InvalidOperation: "cannot cast Array to a different width"
            );
            Box::new(nested::cast_fixed_size_list(downcast(array), |values| {
                cast(values, from_inner, to_inner, options)
            })?)
        },
        #[cfg(feature = "dtype-array")]
        (D::List(from_inner), D::Array(to_inner, width)) => Box::new(
            nested::list_to_fixed_size_list(downcast(array), *width, |values| {
                cast(values, from_inner, to_inner, options)
            })?,
        ),
        #[cfg(feature = "dtype-array")]
        (D::Array(from_inner, _), D::List(to_inner)) => Box::new(nested::fixed_size_list_to_list(
            downcast(array),
            |values| cast(values, from_inner, to_inner, options),
        )?),

        // The bytes of an element are held one per value, which is what makes the two readable as
        // one another.
        (D::List(inner), D::Binary) if matches!(**inner, D::UInt8) => {
            Box::new(nested::list_uint8_to_binview(downcast(array))?)
        },
        (D::Binary, D::List(inner)) if matches!(**inner, D::UInt8) => {
            Box::new(binary::binary_to_list(&binview::view_to_binary(
                downcast::<PlBinaryViewArray>(array),
            )))
        },

        // The bytes of a binary are read as its own elements, so its cast answers for the pair.
        (D::Binary | D::BinaryOffset, D::List(_)) => return Ok(None),
        (_, D::List(_)) => polars_bail!(
            InvalidOperation:
            "casting from {from:?} to list type is not supported\n\
            Hint: Use pl.list(expr) to turn the {from:?} column into a column of single-element lists."
        ),
        _ => return Ok(None),
    };

    Ok(Some(out))
}

/// An array of `length` nulls of `dtype`, which needs no slot per element of a flat type.
fn full_null(dtype: &DataType, length: usize) -> PolarsResult<Box<dyn PlArray>> {
    use DataType as D;

    let dtype = dtype.to_physical();
    Ok(match &dtype {
        D::Null => Box::new(PlNullArray::new(length)),
        D::Boolean => Box::new(PlBooleanArray::new_full_null(length)),
        D::String => Box::new(PlUtf8ViewArray::new_full_null(length)),
        D::Binary => Box::new(PlBinaryViewArray::new_full_null(length)),
        D::BinaryOffset => Box::new(PlBinaryArray::new_full_null(length)),
        D::List(inner) => Box::new(polars_array::PlListArray::new_full_null(
            empty(inner)?,
            length,
        )),
        #[cfg(feature = "dtype-array")]
        D::Array(inner, width) => Box::new(polars_array::PlFixedSizeListArray::new_full_null(
            full_null(inner, *width)?,
            length,
        )),
        #[cfg(feature = "dtype-struct")]
        D::Struct(fields) => Box::new(polars_array::PlStructArray::new_full_null(
            fields
                .iter()
                .map(|field| full_null(field.dtype(), length))
                .collect::<PolarsResult<_>>()?,
            length,
        )),
        dtype => match dtype.to_pl_array_type() {
            PlArrayType::Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
                Box::new(PlPrimitiveArray::<$T>::new_full_null(length)) as Box<dyn PlArray>
            }),
            _ => polars_bail!(InvalidOperation: "cannot cast to {dtype:?}"),
        },
    })
}

/// An empty array of `dtype`, which is what the values of an array of empty lists are.
fn empty(dtype: &DataType) -> PolarsResult<Box<dyn PlArray>> {
    full_null(dtype, 0)
}

/// Whether `dtype` is a number laid out as the number it is.
fn is_plain_numeric(dtype: &DataType) -> bool {
    use DataType as D;
    // The widest integers and the half-precision float are read as the numbers they are only
    // where they are compiled in at all.
    match dtype {
        D::UInt128 => cfg!(feature = "dtype-u128"),
        D::Int128 => cfg!(feature = "dtype-i128"),
        D::Float16 => cfg!(feature = "dtype-f16"),
        dtype => matches!(
            dtype,
            D::UInt8
                | D::UInt16
                | D::UInt32
                | D::UInt64
                | D::Int8
                | D::Int16
                | D::Int32
                | D::Int64
                | D::Float32
                | D::Float64
        ),
    }
}

/// Whether `dtype` is nothing but a name over the physical type its values are laid out as.
fn wraps_its_physical_type(dtype: &DataType) -> bool {
    use DataType as D;
    matches!(
        dtype,
        D::Date | D::Time | D::Datetime(_, _) | D::Duration(_)
    )
}

/// Whether a cast is nothing but a change of the name over the same values.
fn is_retag(from: &DataType, to: &DataType) -> bool {
    // An `i64` holds more than a day's worth of nanoseconds, and the ones that fall outside a day
    // name no time: reading them as one is a range check rather than a re-tag.
    if matches!(to, DataType::Time) {
        return false;
    }

    (wraps_its_physical_type(from) && to == &from.to_physical())
        || (wraps_its_physical_type(to) && from == &to.to_physical())
}

/// Whether the values are cast with `as` rather than a checked conversion, saturating instead of
/// reading as null.
fn casts_with_as(from: &DataType, to: &DataType) -> bool {
    use polars_array::PrimitiveType::*;

    let (PlArrayType::Primitive(from), PlArrayType::Primitive(to)) =
        (from.to_pl_array_type(), to.to_pl_array_type())
    else {
        unreachable!("a plain numeric type is held by a primitive array")
    };

    matches!(
        (from, to),
        (
            UInt8,
            UInt16 | UInt32 | UInt64 | Float16 | Float32 | Float64
        ) | (UInt16, UInt32 | UInt64 | Float16 | Float32 | Float64)
            | (UInt32, UInt64 | Float16 | Float32 | Float64)
            | (UInt64, Float16 | Float32 | Float64)
            | (UInt128, Float16 | Float32 | Float64)
            | (
                Int8,
                Int16 | Int32 | Int64 | Int128 | Float16 | Float32 | Float64
            )
            | (Int16, Int32 | Int64 | Int128 | Float16 | Float32 | Float64)
            | (Int32, Int64 | Int128 | Float16 | Float32 | Float64)
            | (Int64, Float16 | Float64)
            | (Int128, Float16 | Float64)
            | (Float16, Float32 | Float64)
            | (Float32, Float16 | Float64)
            | (Float64, Float16)
    )
}

/// The number of a time unit's steps in a second, which is what its values count.
fn time_unit_multiple(unit: polars_dtype::TimeUnit) -> i64 {
    super::temporal::time_unit_multiple(unit.to_arrow())
}

const SECONDS_IN_DAY: i64 = 86_400;

fn unsupported<T>(from: &DataType, to: &DataType) -> PolarsResult<T> {
    polars_bail!(InvalidOperation: "casting from {from:?} to {to:?} not supported")
}

#[inline]
fn downcast<A: PlArray + 'static>(array: &dyn PlArray) -> &A {
    array
        .as_any()
        .downcast_ref()
        .expect("the data type cast from names the array holding its values")
}

/// Applies `op` to every value of `from`, reading a scalar chunk's one value once.
fn map_values<I, O, F>(from: &PlPrimitiveArray<I>, op: F) -> PlPrimitiveArray<O>
where
    I: arrow::types::NativeType,
    O: arrow::types::NativeType,
    F: Fn(I) -> O,
{
    match from.scalar_value_ignore_validity() {
        Some(value) => PlPrimitiveArray::new_scalar(op(value), from.len())
            .with_validity(from.validity().map(PlBitmap::from)),
        // The values hold a slot per element, so this is the one place the cast writes one too.
        // The shared kernel is `#[inline(never)]` over the element types, which keeps one unrolled
        // loop rather than one per pair of types cast between.
        None => crate::arity::prim_unary_values(from.to_flat().into_owned(), op),
    }
}

/// Applies `op` to every value of `from`, leaving a null wherever it answers `None`.
fn map_values_fallible<I, O, F>(from: &PlPrimitiveArray<I>, op: F) -> PlPrimitiveArray<O>
where
    I: arrow::types::NativeType,
    O: arrow::types::NativeType,
    F: Fn(I) -> Option<O>,
{
    // The one value every element of a scalar chunk reads is cast once, and the answer repeats it
    // in turn.
    if let Some(value) = from.scalar_value_ignore_validity() {
        return match op(value) {
            Some(cast) => PlPrimitiveArray::new_scalar(cast, from.len())
                .with_validity(from.validity().map(PlBitmap::from)),
            None => PlPrimitiveArray::new_full_null(from.len()),
        };
    }

    let values = from.flat_values().unwrap();
    let mut fits = MaskBuilder::with_capacity(values.len());
    let mut out = Vec::with_capacity(values.len());
    for &value in values.iter() {
        let cast = op(value);
        fits.push(cast.is_some());
        out.push(cast.unwrap_or_default());
    }
    let out = PlPrimitiveArray::from_vec(out);
    match fits.finish() {
        // Every value fit, so the mask the array came with is the whole answer.
        None => out.with_validity(from.validity().map(PlBitmap::from)),
        Some(fits) => out.with_validity(Some(and_validity(from.validity(), fits))),
    }
}

/// Applies `op` to the bytes of every element, leaving a null wherever it answers `None`.
fn map_bytes_fallible<'a, O, F>(
    length: usize,
    scalar_value: Option<&[u8]>,
    values: impl Iterator<Item = &'a [u8]>,
    validity: Option<PlBitmapRef<'_>>,
    op: F,
) -> PlPrimitiveArray<O>
where
    O: arrow::types::NativeType,
    F: Fn(&[u8]) -> Option<O>,
{
    // The one value every element of a scalar chunk reads is cast once, and the answer repeats it
    // in turn.
    if let Some(value) = scalar_value {
        return match op(value) {
            Some(cast) => PlPrimitiveArray::new_scalar(cast, length)
                .with_validity(validity.map(PlBitmap::from)),
            None => PlPrimitiveArray::new_full_null(length),
        };
    }

    let mut fits = MaskBuilder::with_capacity(length);
    let mut out = Vec::with_capacity(length);
    for value in values {
        let cast = op(value);
        fits.push(cast.is_some());
        out.push(cast.unwrap_or_default());
    }
    let out = PlPrimitiveArray::from_vec(out);
    match fits.finish() {
        // Every value was read, so the mask the array came with is the whole answer.
        None => out.with_validity(validity.map(PlBitmap::from)),
        Some(fits) => out.with_validity(Some(and_validity(validity, fits))),
    }
}

/// Unsets the mask wherever `keep` does not hold, which is how a narrowing cast reports a miss.
fn mask_where<T, F>(array: &PlPrimitiveArray<T>, keep: F) -> PlPrimitiveArray<T>
where
    T: arrow::types::NativeType,
    F: Fn(T) -> bool,
{
    if let Some(value) = array.scalar_value_ignore_validity() {
        return if keep(value) {
            array.clone()
        } else {
            PlPrimitiveArray::new_full_null(array.len())
        };
    }

    let values = array.flat_values().unwrap();
    let mut fits = MaskBuilder::with_capacity(values.len());
    for &value in values.iter() {
        fits.push(keep(value));
    }
    match fits.finish() {
        None => array.clone(),
        Some(fits) => array
            .clone()
            .with_validity(Some(and_validity(array.validity(), fits))),
    }
}

/// And `mask` into `validity`, which is how a cast reports the values it dropped.
fn and_validity(validity: Option<PlBitmapRef<'_>>, mask: arrow::bitmap::Bitmap) -> PlBitmap {
    // The cast's own mask holds one bit per element, but the array's comes in whichever
    // representation it is in: a chunk that is null throughout, or valid throughout, carries a
    // single bit that settles the `and` without being written out first.
    let length = mask.len();
    let mask = PlBitmapRef::new(&mask, length);

    combine_validities_and(validity, Some(mask))
        .expect("a mask was handed in, so the combination is one too")
}

/// Collects the bit a cast set for each element, answering `None` if it set them all.
struct MaskBuilder {
    builder: arrow::bitmap::BitmapBuilder,
    all_set: bool,
}

impl MaskBuilder {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            builder: arrow::bitmap::BitmapBuilder::with_capacity(capacity),
            all_set: true,
        }
    }

    #[inline]
    fn push(&mut self, bit: bool) {
        self.all_set &= bit;
        self.builder.push(bit);
    }

    fn finish(self) -> Option<arrow::bitmap::Bitmap> {
        (!self.all_set).then(|| self.builder.freeze())
    }
}
