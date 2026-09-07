//! Casting the arrays of `polars-array` without crossing over to Arrow.

use arrow::array::LIST_VALUES_NAME;
use arrow::datatypes::{ArrowDataType, PhysicalType, PrimitiveType, TimeUnit};
use arrow::types::NativeType;
use arrow::with_match_primitive_type;
use polars_array::{
    PlArray, PlArrayType, PlBinaryArray, PlBinaryViewArray, PlBitmap, PlBitmapRef, PlBooleanArray,
    PlFixedSizeBinaryArray, PlFixedSizeListArray, PlListArray, PlNullArray, PlPrimitiveArray,
    PlStructArray, PlUtf8ViewArray,
};
use polars_error::PolarsResult;
use polars_utils::format_pl_smallstr;

use super::CastOptionsImpl;
use crate::comparisons::PlTotalEqKernel;

/// The Arrow type that says how to read the buffers of `array`, which is what it exports as.
pub fn physical_dtype(array: &dyn PlArray) -> ArrowDataType {
    use PlArrayType as A;
    match array.array_type() {
        A::Null => ArrowDataType::Null,
        A::Boolean => ArrowDataType::Boolean,
        A::Primitive(primitive) => ArrowDataType::from(primitive),
        A::Binary => ArrowDataType::LargeBinary,
        A::BinaryView => ArrowDataType::BinaryView,
        A::Utf8View => ArrowDataType::Utf8View,
        A::FixedSizeBinary => {
            ArrowDataType::FixedSizeBinary(downcast::<PlFixedSizeBinaryArray>(array).width())
        },
        A::List => {
            let array = downcast::<PlListArray>(array);
            ArrowDataType::LargeList(Box::new(arrow::datatypes::Field::new(
                LIST_VALUES_NAME,
                physical_dtype(array.values()),
                true,
            )))
        },
        A::FixedSizeList => {
            let array = downcast::<PlFixedSizeListArray>(array);
            ArrowDataType::FixedSizeList(
                Box::new(arrow::datatypes::Field::new(
                    LIST_VALUES_NAME,
                    physical_dtype(array.values()),
                    true,
                )),
                array.width(),
            )
        },
        A::Struct => {
            let array = downcast::<PlStructArray>(array);
            let fields = array
                .fields()
                .iter()
                .enumerate()
                .map(|(i, field)| {
                    arrow::datatypes::Field::new(
                        format_pl_smallstr!("{i}"),
                        physical_dtype(&**field),
                        true,
                    )
                })
                .collect();
            ArrowDataType::Struct(fields)
        },
        array_type @ A::Object { .. } => {
            unimplemented!("polars-compute: {array_type:?} has no Arrow type to cast on")
        },
    }
}

#[inline]
fn downcast<A: PlArray + 'static>(array: &dyn PlArray) -> &A {
    array
        .as_any()
        .downcast_ref()
        .expect("the array type dispatched on names the array")
}

/// Casts `array` from `from_type` to `to_type`, or `None` if the pair belongs to the Arrow kernels.
pub(super) fn cast_native(
    array: &dyn PlArray,
    from_type: &ArrowDataType,
    to_type: &ArrowDataType,
    options: CastOptionsImpl,
) -> Option<PolarsResult<Box<dyn PlArray>>> {
    use ArrowDataType::*;

    // A cast that changes nothing but the logical type reads the same values, and an array here
    // holds no logical type to change: the array *is* the answer.
    if from_type == to_type || is_retag(from_type, to_type) {
        return Some(Ok(array.to_boxed()));
    }

    // A dictionary is the one array of the Arrow set no array here holds.
    if matches!(from_type, Dictionary(..)) || matches!(to_type, Dictionary(..)) {
        return None;
    }

    match (from_type, to_type) {
        // Null on either side reads as null everywhere, which needs no slot per element.
        (Null, _) => full_null(to_type, array.len()).map(Ok),
        (_, Null) => Some(Ok(Box::new(PlNullArray::new(array.len())))),

        (Boolean, _) if is_plain_numeric(to_type) => Some(Ok(with_match_primitive_type!(
            primitive_of(to_type),
            |$T| Box::new(boolean_to_primitive::<$T>(downcast(array))) as Box<dyn PlArray>
        ))),

        (_, Boolean) if is_plain_numeric(from_type) => Some(Ok(with_match_primitive_type!(
            primitive_of(from_type),
            |$T| Box::new(primitive_to_boolean::<$T>(downcast(array))) as Box<dyn PlArray>
        ))),

        // Not a conversion but a range check: `Time64(ns)` holds a day's worth of nanoseconds, so
        // an `i64` outside that range names no time and reads as null.
        (Int64, Time64(TimeUnit::Nanosecond)) => {
            const NANOS_PER_DAY: i64 = 86_400_000_000_000;
            let array: &PlPrimitiveArray<i64> = downcast(array);
            Some(Ok(Box::new(mask_where(array, |v| {
                (0..NANOS_PER_DAY).contains(&v)
            }))))
        },

        _ if is_plain_numeric(from_type) && is_plain_numeric(to_type) => {
            let wrapped =
                options.wrapped || casts_with_as(primitive_of(from_type), primitive_of(to_type));
            Some(Ok(
                with_match_primitive_type!(primitive_of(from_type), |$I| {
                    let from: &PlPrimitiveArray<$I> = downcast(array);
                    with_match_primitive_type!(primitive_of(to_type), |$O| {
                        Box::new(numeric_to_numeric::<$I, $O>(from, wrapped)) as Box<dyn PlArray>
                    })
                }),
            ))
        },

        _ => None,
    }
}

/// Whether `dtype` is a number laid out as the number it is.
fn is_plain_numeric(dtype: &ArrowDataType) -> bool {
    use ArrowDataType::*;
    dtype.is_numeric()
        && !matches!(
            dtype,
            Decimal(..) | Decimal32(..) | Decimal64(..) | Decimal256(..)
        )
}

/// The element type of a numeric Arrow type.
fn primitive_of(dtype: &ArrowDataType) -> PrimitiveType {
    match dtype.to_physical_type() {
        PhysicalType::Primitive(primitive) => primitive,
        physical => unreachable!("a numeric type is primitive, got {physical:?}"),
    }
}

/// Whether a cast is nothing but a change of logical type over the same values.
fn is_retag(from_type: &ArrowDataType, to_type: &ArrowDataType) -> bool {
    use ArrowDataType::*;
    matches!(
        (from_type, to_type),
        (Int32, Date32)
            | (Date32, Int32)
            | (Time32(_), Int32)
            | (Date64, Int64)
            | (Time64(_), Int64)
            | (Timestamp(..), Int64)
            | (Int64, Timestamp(..))
            | (Int64, Duration(_))
            | (Duration(_), Int64)
    )
}

/// Whether the Arrow kernels cast this pair with `as` rather than a checked conversion.
fn casts_with_as(from: PrimitiveType, to: PrimitiveType) -> bool {
    use PrimitiveType::*;
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

/// An array of `length` nulls held by the array type `dtype` names, needing no slot per element.
fn full_null(dtype: &ArrowDataType, length: usize) -> Option<Box<dyn PlArray>> {
    Some(match dtype.to_physical_type() {
        PhysicalType::Null => Box::new(PlNullArray::new(length)),
        PhysicalType::Boolean => Box::new(PlBooleanArray::new_full_null(length)),
        PhysicalType::Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            Box::new(PlPrimitiveArray::<$T>::new_full_null(length)) as Box<dyn PlArray>
        }),
        PhysicalType::Binary
        | PhysicalType::LargeBinary
        | PhysicalType::Utf8
        | PhysicalType::LargeUtf8 => Box::new(PlBinaryArray::new_full_null(length)),
        PhysicalType::BinaryView => Box::new(PlBinaryViewArray::new_full_null(length)),
        PhysicalType::Utf8View => Box::new(PlUtf8ViewArray::new_full_null(length)),
        // A nested shape needs its children built too, which the Arrow kernel already does.
        _ => return None,
    })
}

/// Casts the values of `from` to `O`, leaving a null where a value does not fit.
fn numeric_to_numeric<I, O>(from: &PlPrimitiveArray<I>, wrapped: bool) -> PlPrimitiveArray<O>
where
    I: NativeType + num_traits::NumCast + num_traits::AsPrimitive<O>,
    O: NativeType + num_traits::NumCast,
{
    // A wrapping cast answers for every value, so the mask is the one the array came with.
    if wrapped {
        return map_values(from, num_traits::AsPrimitive::<O>::as_);
    }

    // The one value every element of a scalar chunk reads is cast once, and the answer repeats it
    // in turn.
    if let Some(value) = from.scalar_values() {
        return match num_traits::cast::cast::<I, O>(value) {
            Some(cast) => PlPrimitiveArray::new_scalar(cast, from.len())
                .with_validity(from.validity().map(PlBitmap::from)),
            None => PlPrimitiveArray::new_full_null(from.len()),
        };
    }

    let mut fits = MaskBuilder::with_capacity(from.len());
    let mut out = Vec::with_capacity(from.len());
    for &value in from.flat_values().unwrap().iter() {
        let cast = num_traits::cast::cast::<I, O>(value);
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

/// Applies `op` to every value of `from`, reading a scalar chunk's one value once.
fn map_values<I, O, F>(from: &PlPrimitiveArray<I>, op: F) -> PlPrimitiveArray<O>
where
    I: NativeType,
    O: NativeType,
    F: Fn(I) -> O,
{
    match from.scalar_values() {
        Some(value) => PlPrimitiveArray::new_scalar(op(value), from.len())
            .with_validity(from.validity().map(PlBitmap::from)),
        // The values hold a slot per element, so this is the one place the cast writes one too.
        // The shared kernel is `#[inline(never)]` over the element types, which keeps one unrolled
        // loop rather than one per pair of types cast between.
        None => crate::arity::prim_unary_values(from.as_flat().unwrap().clone(), op),
    }
}

/// Unsets the mask wherever `keep` does not hold, which is how a narrowing cast reports a miss.
fn mask_where<T, F>(array: &PlPrimitiveArray<T>, keep: F) -> PlPrimitiveArray<T>
where
    T: NativeType,
    F: Fn(T) -> bool,
{
    if let Some(value) = array.scalar_values() {
        return if keep(value) {
            array.clone()
        } else {
            PlPrimitiveArray::new_full_null(array.len())
        };
    }

    let mut fits = MaskBuilder::with_capacity(array.len());
    for &value in array.flat_values().unwrap().iter() {
        fits.push(keep(value));
    }
    match fits.finish() {
        None => array.clone(),
        Some(fits) => array
            .clone()
            .with_validity(Some(and_validity(array.validity(), fits))),
    }
}

fn boolean_to_primitive<T>(from: &PlBooleanArray) -> PlPrimitiveArray<T>
where
    T: NativeType + num_traits::One,
{
    let value_of = |set: bool| if set { T::one() } else { T::default() };
    let values = match from.scalar_values() {
        Some(value) => PlPrimitiveArray::new_scalar(value_of(value), from.len()),
        None => {
            let out: Vec<T> = from.flat_values().unwrap().iter().map(value_of).collect();
            PlPrimitiveArray::from_vec(out)
        },
    };
    values.with_validity(from.validity().map(PlBitmap::from))
}

fn primitive_to_boolean<T>(from: &PlPrimitiveArray<T>) -> PlBooleanArray
where
    T: NativeType,
    PlPrimitiveArray<T>: PlTotalEqKernel<Scalar = T>,
{
    // The comparison kernel answers over the representation the values are in, so a chunk that
    // repeats one value is compared once and the answer repeats in turn.
    let values = from.tot_ne_kernel_broadcast(&T::default());
    PlBooleanArray::from_pl_bitmap(values).with_validity(from.validity().map(PlBitmap::from))
}

/// Ands `mask` into `validity`, which is how a cast reports the values it dropped.
fn and_validity(validity: Option<PlBitmapRef<'_>>, mask: arrow::bitmap::Bitmap) -> PlBitmap {
    let length = mask.len();
    match validity {
        None => PlBitmap::new(mask, length),
        Some(validity) => {
            // A mask of one element reads as scalar behind `flat_bitmap`, so the bits are taken
            // off the flattened mask itself.
            let validity = PlBitmap::from(validity)
                .to_flat()
                .into_owned()
                .into_inner()
                .0;
            PlBitmap::new(arrow::bitmap::and(&validity, &mask), length)
        },
    }
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
