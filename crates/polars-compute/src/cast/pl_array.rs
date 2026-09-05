//! Casting the arrays of `polars-array` without crossing over to Arrow.
//!
//! A cast is dispatched on a *pair* of [`ArrowDataType`]s, which is what the Arrow kernels in the
//! parent module read off the two arrays. An array of `polars-array` carries no type of its own,
//! so the pair is passed in instead: `from_type` says how to read the array's buffers — which is
//! all a [`PlBinaryArray`](polars_array::PlBinaryArray) needs to be told apart as bytes or as
//! UTF-8 — and `to_type` says what to build.
//!
//! The arms here read and write the arrays of `polars-array` directly, so a chunk is neither
//! written out nor handed over on the way in or out, and a chunk that repeats one value stays one
//! that repeats one value. [`cast_native`] answers `None` for a pair it does not hold a kernel
//! for, which the parent module then casts over the Arrow arrays.

use arrow::array::LIST_VALUES_NAME;
use arrow::datatypes::{ArrowDataType, PhysicalType, PrimitiveType, TimeUnit};
use arrow::types::NativeType;
use arrow::with_match_primitive_type;
use polars_array::{
    ArrayRepr, PlArray, PlArrayType, PlBinaryArray, PlBinaryViewArray, PlBitmap, PlBitmapRef,
    PlBooleanArray, PlFixedSizeBinaryArray, PlFixedSizeListArray, PlListArray, PlNullArray,
    PlPrimitiveArray, PlStructArray, PlUtf8ViewArray,
};
use polars_error::PolarsResult;
use polars_utils::format_pl_smallstr;

use super::CastOptionsImpl;
use crate::comparisons::PlTotalEqKernel;

/// The Arrow type that says how to read the buffers of `array`, which is the type it crosses over
/// to Arrow as — see [`polars_array::arrow::export`].
///
/// This is the *physical* type: an array of `polars-array` holds no logical type, so an `i64` of
/// nanoseconds since the epoch reads as [`Int64`](ArrowDataType::Int64), never as a
/// [`Timestamp`](ArrowDataType::Timestamp), and the bytes of a string read as
/// [`LargeBinary`](ArrowDataType::LargeBinary), never as [`LargeUtf8`](ArrowDataType::LargeUtf8). A
/// caller holding the values of a logical type passes its own `from_type` to
/// [`cast_chunk_from`](super::cast_chunk_from) rather than letting it be read off the array here.
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

/// Casts `array` from `from_type` to `to_type` over the arrays of `polars-array`, or answers `None`
/// if this pair has no kernel here and belongs to the Arrow ones.
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
///
/// A decimal is *not*: it is an integer scaled by a power of ten, so converting one is a
/// rescaling, not the conversion of the integer under it — which is what makes
/// [`is_numeric`](ArrowDataType::is_numeric), which counts decimals in, the wrong question here.
/// Those go to the scale-aware kernels of the parent module.
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

/// Whether a cast between these two types is nothing but a change of the logical type over the
/// same values, which an array of `polars-array` — holding no logical type — answers by handing
/// itself back.
///
/// These are exactly the pairs the Arrow kernels answer with `primitive_to_same_primitive`, which
/// hands the values buffer and the mask over under a new type. A pair that converts anything — a
/// width, a unit, a scale — is not one of them, and neither is a pair the Arrow kernels reject:
/// this list is read off that dispatch rather than derived from what would *happen* to line up, so
/// a cast polars does not support does not start succeeding here.
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
///
/// This is not only how fast the cast is: `as` saturates where a checked conversion answers that
/// the value does not fit and leaves a null, so `1e30f32` reads as `inf` under one and as null
/// under the other. The pairs are read off the `as_options` arms of the Arrow dispatch so that a
/// value casts to the same thing whichever kernel answers it.
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

/// An array of `length` nulls held by the array type `dtype` names, which needs no slot per
/// element.
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

/// Casts the values of `from` to `O`, leaving a null where a value does not fit unless the cast
/// wraps.
fn numeric_to_numeric<I, O>(from: &PlPrimitiveArray<I>, wrapped: bool) -> PlPrimitiveArray<O>
where
    I: NativeType + num_traits::NumCast + num_traits::AsPrimitive<O>,
    O: NativeType + num_traits::NumCast,
{
    // A wrapping cast answers for every value, so the mask is the one the array came with.
    if wrapped {
        return map_values(from, num_traits::AsPrimitive::<O>::as_);
    }

    match from.values_repr() {
        // The one value every element reads is cast once, and the answer repeats it in turn.
        ArrayRepr::Scalar(value) => match num_traits::cast::cast::<I, O>(value) {
            Some(cast) => PlPrimitiveArray::new_scalar(cast, from.len())
                .with_validity(from.validity().map(PlBitmap::from)),
            None => PlPrimitiveArray::new_full_null(from.len()),
        },
        ArrayRepr::Flat(values) => {
            let mut fits = MaskBuilder::with_capacity(from.len());
            let mut out = Vec::with_capacity(from.len());
            for &value in values.iter() {
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
        },
    }
}

/// Applies `op` to every value of `from`, reading the one value of a chunk that repeats one value
/// once and leaving the answer repeating it in turn.
fn map_values<I, O, F>(from: &PlPrimitiveArray<I>, op: F) -> PlPrimitiveArray<O>
where
    I: NativeType,
    O: NativeType,
    F: Fn(I) -> O,
{
    match from.values_repr() {
        ArrayRepr::Scalar(value) => PlPrimitiveArray::new_scalar(op(value), from.len())
            .with_validity(from.validity().map(PlBitmap::from)),
        // The values hold a slot per element, so this is the one place the cast writes one too.
        // The shared kernel is `#[inline(never)]` over the element types, which keeps one unrolled
        // loop rather than one per pair of types cast between.
        ArrayRepr::Flat(_) => crate::arity::prim_unary_values(from.to_flat().into_owned(), op),
    }
}

/// Unsets the mask wherever `keep` does not hold, which is how a cast into a type holding fewer
/// values than its representation reports the ones with none.
fn mask_where<T, F>(array: &PlPrimitiveArray<T>, keep: F) -> PlPrimitiveArray<T>
where
    T: NativeType,
    F: Fn(T) -> bool,
{
    match array.values_repr() {
        ArrayRepr::Scalar(value) => {
            if keep(value) {
                array.clone()
            } else {
                PlPrimitiveArray::new_full_null(array.len())
            }
        },
        ArrayRepr::Flat(values) => {
            let mut fits = MaskBuilder::with_capacity(array.len());
            for &value in values.iter() {
                fits.push(keep(value));
            }
            match fits.finish() {
                None => array.clone(),
                Some(fits) => array
                    .clone()
                    .with_validity(Some(and_validity(array.validity(), fits))),
            }
        },
    }
}

fn boolean_to_primitive<T>(from: &PlBooleanArray) -> PlPrimitiveArray<T>
where
    T: NativeType + num_traits::One,
{
    let value_of = |set: bool| if set { T::one() } else { T::default() };
    match from.values_repr() {
        ArrayRepr::Scalar(value) => PlPrimitiveArray::new_scalar(value_of(value), from.len())
            .with_validity(from.validity().map(PlBitmap::from)),
        ArrayRepr::Flat(values) => {
            let out: Vec<T> = values.iter().map(value_of).collect();
            PlPrimitiveArray::from_vec(out).with_validity(from.validity().map(PlBitmap::from))
        },
    }
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

/// Ands `mask` into `validity`, which is how a cast that dropped values reports them alongside the
/// nulls the array already held.
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

/// Collects the bit a cast set for each element, answering `None` if it set them all — the common
/// case, and the one that leaves the array's own mask untouched.
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

#[cfg(test)]
mod tests {
    use polars_array::arrow::export;

    use super::*;

    /// Every array of `polars-array` that this module reads a type off crosses over to Arrow
    /// carrying that very type.
    ///
    /// This is the property the whole dispatch rests on: [`physical_dtype`] stands in for the tag
    /// [`export::to_arrow`] stamps, so a cast reading a chunk's type here answers what the Arrow
    /// kernels would have been handed. A representation that starts carrying its own type, or an
    /// export that starts naming its fields differently, has to be answered in both places at
    /// once — and this test is what says so.
    #[test]
    fn a_derived_type_is_the_one_the_array_crosses_over_as() {
        let arrays: Vec<Box<dyn PlArray>> = vec![
            Box::new(PlNullArray::new(3)),
            Box::new(PlBooleanArray::new_scalar(true, 3)),
            Box::new(PlPrimitiveArray::from_vec(vec![1i8, 2, 3])),
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
            Box::new(PlPrimitiveArray::from_vec(vec![1i64, 2, 3])),
            Box::new(PlPrimitiveArray::from_vec(vec![1u128, 2, 3])),
            Box::new(PlPrimitiveArray::from_vec(vec![1.5f64, 2.5, 3.5])),
            Box::new(PlPrimitiveArray::new_scalar(7i32, 3)),
            Box::new(PlBinaryArray::new_full_null(3)),
            Box::new(PlBinaryViewArray::new_full_null(3)),
            Box::new(PlUtf8ViewArray::new_full_null(3)),
        ];

        for array in &arrays {
            assert_eq!(
                &physical_dtype(&**array),
                export::to_arrow(&**array).dtype(),
                "{:?} reads as a type it does not cross over as",
                array.array_type(),
            );
        }
    }

    /// A cast that only changes the logical type hands the array straight back, which is what a
    /// representation carrying no logical type makes it.
    #[test]
    fn a_change_of_logical_type_alone_is_the_array_itself() {
        use ArrowDataType::*;

        // Same values, different type over them.
        assert!(is_retag(&Int32, &Date32));
        assert!(is_retag(&Date32, &Int32));
        assert!(is_retag(&Int64, &Timestamp(TimeUnit::Microsecond, None)));
        assert!(is_retag(&Timestamp(TimeUnit::Microsecond, None), &Int64));
        assert!(is_retag(&Int64, &Duration(TimeUnit::Nanosecond)));

        // A change of unit is a change of what the values mean, so it is real work.
        assert!(!is_retag(
            &Timestamp(TimeUnit::Second, None),
            &Timestamp(TimeUnit::Nanosecond, None),
        ));
        assert!(!is_retag(
            &Duration(TimeUnit::Millisecond),
            &Duration(TimeUnit::Nanosecond),
        ));

        // A change of width is a conversion, however close the two types read.
        assert!(!is_retag(&Int32, &Int64));
        // A decimal's scale is part of what its values mean, so it is never a bare re-tag.
        assert!(!is_retag(&Int128, &Decimal(10, 2)));
        assert!(!is_retag(&Decimal(10, 2), &Int128));
        assert!(!is_retag(&Decimal(10, 2), &Decimal(10, 4)));
        assert!(!is_retag(&Date32, &Int64));
        assert!(!is_retag(&Int64, &Float64));
    }

    /// A chunk that repeats one value is cast once and the answer repeats it in turn, rather than
    /// the chunk being written out one slot per element to be cast.
    #[test]
    fn a_repeated_value_stays_repeated_across_a_cast() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, 100);
        let out = cast_native(
            &scalar,
            &ArrowDataType::Int32,
            &ArrowDataType::Float64,
            CastOptionsImpl::default(),
        )
        .expect("a numeric cast is answered here")
        .unwrap();
        assert!(out.is_scalar(), "{out:?} was written out");
        assert_eq!(out.len(), 100);

        // ... and so does the boolean a comparison against zero reads off it.
        let out = cast_native(
            &scalar,
            &ArrowDataType::Int32,
            &ArrowDataType::Boolean,
            CastOptionsImpl::default(),
        )
        .expect("a cast to boolean is answered here")
        .unwrap();
        assert!(out.is_scalar(), "{out:?} was written out");
    }

    /// A pair the Arrow kernels cast with `as` casts with `as` here too, which is what a value
    /// that does not fit reads as: `as` saturates where a checked conversion leaves a null.
    ///
    /// Getting this wrong is silent — the cast still answers, with different values — so the
    /// pairs are pinned here rather than left to whichever conversion looks right.
    #[test]
    fn a_pair_the_arrow_kernels_saturate_saturates_here() {
        use PrimitiveType::*;

        // Read off the `as_options` arms of the Arrow dispatch.
        assert!(casts_with_as(Int32, Int64), "a widening int is not checked");
        assert!(
            casts_with_as(Float32, Float16),
            "a narrowing float saturates"
        );
        assert!(casts_with_as(Float64, Float16));
        assert!(casts_with_as(UInt8, Float64));

        // ... and the arms that pass the caller's own options through.
        assert!(!casts_with_as(Float64, Float32));
        assert!(!casts_with_as(Float64, Int64));
        assert!(!casts_with_as(Int64, Int32));
        assert!(!casts_with_as(Int64, Float32));
        assert!(!casts_with_as(UInt8, Int8));

        // A float too large for the narrower one saturates to infinity rather than reading as
        // null, even though the caller did not ask for a wrapping cast.
        let array = PlPrimitiveArray::from_vec(vec![1.0e30f32, 1.0]);
        let out = cast_native(
            &array,
            &ArrowDataType::Float32,
            &ArrowDataType::Float16,
            CastOptionsImpl::default(),
        )
        .expect("a numeric cast is answered here")
        .unwrap();
        assert_eq!(
            out.null_count(),
            0,
            "{out:?} read a saturating cast as null"
        );
    }

    /// A value the target type cannot hold reads as null, and the count is read off the answer.
    #[test]
    fn a_value_that_does_not_fit_reads_as_null() {
        let array = PlPrimitiveArray::from_vec(vec![1i32, 300, -5]);
        let out = cast_native(
            &array,
            &ArrowDataType::Int32,
            &ArrowDataType::UInt8,
            CastOptionsImpl::default(),
        )
        .expect("a numeric cast is answered here")
        .unwrap();
        assert_eq!(out.null_count(), 2, "300 and -5 have no u8 to be");

        // A wrapping cast has an answer for every value, so it drops none of them.
        let out = cast_native(
            &array,
            &ArrowDataType::Int32,
            &ArrowDataType::UInt8,
            CastOptionsImpl::unchecked(),
        )
        .expect("a numeric cast is answered here")
        .unwrap();
        assert_eq!(out.null_count(), 0);
    }
}
