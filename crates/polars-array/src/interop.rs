//! Conversion to and from the Arrow arrays of `polars-arrow`.
//!
//! The arrays of this crate are built on the same [`Buffer`] and [`Bitmap`] as their Arrow
//! counterparts and lay their elements out the same way, so an Arrow array is imported by handing
//! its backing buffers over: [`from_arrow`] is `O(1)` for every array but the ones whose offsets
//! are 32 bits wide, which are widened to the 64-bit offsets a [`PlBinaryArray`] and a
//! [`PlListArray`] hold.
//!
//! What does not carry over is the logical type. An Arrow array names one and the arrays here do
//! not, so importing drops it — a [`Utf8ViewArray`] and a [`BinaryViewArray`] both import as a
//! [`PlBinaryViewArray`], and a `Date32` array as a [`PlPrimitiveArray<i32>`] — and [`to_arrow`]
//! takes the [`ArrowDataType`] to export as, since there is nothing left in the array to derive it
//! from. It is the caller that remembers which logical type the physical array stands for.
//!
//! # UTF-8
//!
//! Nothing in this crate says that the bytes of a [`PlBinaryArray`] or a [`PlBinaryViewArray`] are
//! a string, and nothing in it ever validates that they are. Exporting one as an Arrow array that
//! does say so — a [`Utf8Array`] or a [`Utf8ViewArray`] — therefore has to establish it:
//! [`to_arrow`] walks the elements and errors if they are not valid UTF-8, and
//! [`to_arrow_unchecked`] is the unsafe `O(1)` counterpart for a caller that already knows they
//! are. The two are the same function for every other Arrow data type.
//!
//! # The scalar representation
//!
//! An Arrow array has no counterpart of the scalar representation — see [`crate::broadcast`] — so
//! there is nothing to export a scalar array *as*: [`to_arrow`] writes one out with
//! [`to_flat`](crate::PlPrimitiveArray::to_flat), which is `O(len)` in both time and memory. An
//! array whose length is unbounded by its memory use does not stay that way across the export.
//!
//! Importing never produces a scalar array: an Arrow array holds one slot per element by
//! construction, so what comes back out of [`from_arrow`] is always flat.
//!
//! # Arrays with no counterpart
//!
//! A dictionary, union or map array has no counterpart in this crate, and neither has an Arrow
//! array of an element type that is not a [`NativeType`]. Importing one errors rather than
//! encoding it as something else; decoding it into an array that does have a counterpart is a
//! decision for the caller.
//!
//! # Example
//! ```
//! use arrow::array::{Array, Int32Array, Utf8ViewArray};
//! use arrow::datatypes::ArrowDataType;
//! use polars_array::interop::{from_arrow, to_arrow};
//! use polars_array::{PlBinaryViewArray, PlPrimitiveArray};
//!
//! // Importing hands the backing buffers over, and drops the logical type.
//! let arrow = Int32Array::from_slice([1, 2, 3]).to(ArrowDataType::Date32);
//! let array = from_arrow(&arrow).unwrap();
//! assert_eq!(
//!     array.as_any().downcast_ref::<PlPrimitiveArray<i32>>().unwrap().value(2),
//!     3,
//! );
//!
//! // Exporting takes the logical type back, since the array no longer carries one.
//! let exported = to_arrow(&*array, ArrowDataType::Date32).unwrap();
//! assert_eq!(exported.dtype(), &ArrowDataType::Date32);
//! assert_eq!(&exported, &(Box::new(arrow) as Box<dyn Array>));
//!
//! // The bytes of a `PlBinaryViewArray` are a string only if the data type says they are.
//! let array = PlBinaryViewArray::from_values_iter([b"foo".as_slice()]);
//! let exported = to_arrow(&array, ArrowDataType::Utf8View).unwrap();
//! assert_eq!(exported.as_any().downcast_ref::<Utf8ViewArray>().unwrap().value(0), "foo");
//! ```

use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeBinaryArray, FixedSizeListArray,
    ListArray, NullArray, PrimitiveArray, StructArray, Utf8Array, Utf8ViewArray, View,
};
use arrow::bitmap::Bitmap;
use arrow::datatypes::{ArrowDataType, PhysicalType, PrimitiveType};
use arrow::offset::OffsetsBuffer;
use arrow::types::{NativeType, days_ms, i256, months_days_ns};
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_bail, polars_ensure};
use polars_utils::float16::pf16;

use crate::array::PlArray;
use crate::{
    PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlFixedSizeBinaryArray, PlFixedSizeListArray,
    PlListArray, PlNullArray, PlPrimitiveArray, PlStructArray, with_match_pl_primitive_array_type,
};

/// Imports an Arrow array as the array of this crate that holds the same elements.
///
/// The logical type of `array` is dropped: what comes back is the physical array underneath it,
/// which is the same for every logical type over one physical representation. It is always
/// [`flat`](crate::broadcast) — an Arrow array holds one slot per element — and it shares its
/// backing buffers with `array`, so this is `O(1)` except for the offsets of a 32-bit-offset array,
/// which are widened.
///
/// # Errors
/// This function errors if `array` is a dictionary, union or map array, or if its elements are of
/// a type that no array of this crate is taken over. See the [module docs](self).
pub fn from_arrow(array: &dyn Array) -> PolarsResult<Box<dyn PlArray>> {
    let length = array.len();
    let validity = array.validity().cloned();

    Ok(match array.dtype().to_physical_type() {
        PhysicalType::Null => Box::new(PlNullArray::new(length)),

        PhysicalType::Boolean => {
            let array = downcast_arrow::<BooleanArray>(array);
            // SAFETY: the values and the validity mask of an Arrow boolean array hold one bit per
            // element, which is what makes them flat here.
            Box::new(unsafe {
                PlBooleanArray::new_unchecked(array.values().clone(), length, validity)
            })
        },

        PhysicalType::Primitive(primitive) => {
            return primitive_from_arrow(array, primitive, length, validity);
        },

        PhysicalType::Binary => {
            let array = downcast_arrow::<BinaryArray<i32>>(array);
            binary_from_arrow(array.values().clone(), array.offsets(), length, validity)
        },
        PhysicalType::LargeBinary => {
            let array = downcast_arrow::<BinaryArray<i64>>(array);
            binary_from_arrow(array.values().clone(), array.offsets(), length, validity)
        },
        PhysicalType::Utf8 => {
            let array = downcast_arrow::<Utf8Array<i32>>(array);
            binary_from_arrow(array.values().clone(), array.offsets(), length, validity)
        },
        PhysicalType::LargeUtf8 => {
            let array = downcast_arrow::<Utf8Array<i64>>(array);
            binary_from_arrow(array.values().clone(), array.offsets(), length, validity)
        },

        PhysicalType::BinaryView => {
            let array = downcast_arrow::<BinaryViewArray>(array);
            view_from_arrow(
                array.views().clone(),
                array.data_buffers().clone(),
                length,
                validity,
            )
        },
        PhysicalType::Utf8View => {
            let array = downcast_arrow::<Utf8ViewArray>(array);
            view_from_arrow(
                array.views().clone(),
                array.data_buffers().clone(),
                length,
                validity,
            )
        },

        PhysicalType::FixedSizeBinary => {
            let array = downcast_arrow::<FixedSizeBinaryArray>(array);
            // SAFETY: the values of an Arrow fixed size binary array hold `size` bytes per element,
            // and its validity mask one bit per element, which is what makes them flat here.
            Box::new(unsafe {
                PlFixedSizeBinaryArray::new_unchecked(
                    array.values().clone(),
                    array.size(),
                    length,
                    validity,
                )
            })
        },

        PhysicalType::List => {
            let array = downcast_arrow::<ListArray<i32>>(array);
            list_from_arrow(&**array.values(), array.offsets(), length, validity)?
        },
        PhysicalType::LargeList => {
            let array = downcast_arrow::<ListArray<i64>>(array);
            list_from_arrow(&**array.values(), array.offsets(), length, validity)?
        },

        PhysicalType::FixedSizeList => {
            let array = downcast_arrow::<FixedSizeListArray>(array);
            let values = from_arrow(&**array.values())?;
            // SAFETY: the values of an Arrow fixed size list array hold `size` elements per
            // element, and its validity mask one bit per element, which is what makes them flat
            // here.
            Box::new(unsafe {
                PlFixedSizeListArray::new_unchecked(values, array.size(), length, validity)
            })
        },

        PhysicalType::Struct => {
            let array = downcast_arrow::<StructArray>(array);
            let fields = array
                .values()
                .iter()
                .map(|field| from_arrow(&**field))
                .collect::<PolarsResult<Vec<_>>>()?;
            // SAFETY: every field of an Arrow struct array has as many elements as the array, and
            // its validity mask holds one bit per element, which is what makes it flat here.
            Box::new(unsafe { PlStructArray::new_unchecked(fields, length, validity) })
        },

        physical @ (PhysicalType::Dictionary(_) | PhysicalType::Union | PhysicalType::Map) => {
            polars_bail!(
                ComputeError:
                "cannot import an arrow array of physical type {physical:?}: no array of \
                 polars-array holds its elements",
            )
        },
    })
}

/// Exports an array of this crate as the Arrow array of `dtype` that holds the same elements.
///
/// The array carries no logical type of its own, so `dtype` is what the result is given; it has to
/// be a data type of the physical representation `array` is in. A scalar array is written out —
/// see the [module docs](self) — so this is `O(len)` unless every backing buffer is already flat.
///
/// # Errors
/// This function errors if the physical type of `dtype` is not the one of `array`, if `dtype` is
/// a string type and the elements of `array` are not valid UTF-8, or if `dtype` has 32-bit offsets
/// that the elements of `array` overflow. [`to_arrow_unchecked`] is the counterpart that trusts
/// the caller about UTF-8.
pub fn to_arrow(array: &dyn PlArray, dtype: ArrowDataType) -> PolarsResult<Box<dyn Array>> {
    to_arrow_impl(array, dtype, true)
}

/// Exports an array of this crate as the Arrow array of `dtype` that holds the same elements,
/// without checking that they are valid UTF-8.
///
/// This is [`to_arrow`] with the one check that is `O(bytes)` left out, and it is the same function
/// for a `dtype` that is not a string type. Every other error [`to_arrow`] reports is still
/// reported here.
///
/// # Safety
/// If `dtype` is [`Utf8`](ArrowDataType::Utf8), [`LargeUtf8`](ArrowDataType::LargeUtf8) or
/// [`Utf8View`](ArrowDataType::Utf8View), every element of `array` — including the ones under a
/// null — must be valid UTF-8.
pub unsafe fn to_arrow_unchecked(
    array: &dyn PlArray,
    dtype: ArrowDataType,
) -> PolarsResult<Box<dyn Array>> {
    to_arrow_impl(array, dtype, false)
}

/// The body of [`to_arrow`] and [`to_arrow_unchecked`], which differ only in `validate_utf8`.
fn to_arrow_impl(
    array: &dyn PlArray,
    dtype: ArrowDataType,
    validate_utf8: bool,
) -> PolarsResult<Box<dyn Array>> {
    let length = array.len();
    let physical = dtype.to_physical_type();

    Ok(match physical {
        PhysicalType::Null => {
            downcast_pl::<PlNullArray>(array, &dtype)?;
            Box::new(NullArray::new(dtype, length))
        },

        PhysicalType::Boolean => {
            let (values, validity) = downcast_pl::<PlBooleanArray>(array, &dtype)?
                .to_flat()
                .into_inner();
            Box::new(BooleanArray::new(dtype, values, validity))
        },

        PhysicalType::Primitive(primitive) => primitive_to_arrow(array, primitive, &dtype)?,

        PhysicalType::Binary
        | PhysicalType::LargeBinary
        | PhysicalType::Utf8
        | PhysicalType::LargeUtf8 => {
            let (values, offsets, validity) = downcast_pl::<PlBinaryArray>(array, &dtype)?
                .to_flat()
                .into_inner();

            match physical {
                PhysicalType::Binary => Box::new(BinaryArray::new(
                    dtype,
                    narrow_offsets(&offsets)?,
                    values,
                    validity,
                )),
                PhysicalType::LargeBinary => Box::new(BinaryArray::new(
                    dtype,
                    widen_offsets(offsets),
                    values,
                    validity,
                )),
                PhysicalType::Utf8 => {
                    let offsets = narrow_offsets(&offsets)?;
                    if validate_utf8 {
                        Box::new(Utf8Array::try_new(dtype, offsets, values, validity)?)
                    } else {
                        // SAFETY: the caller of `to_arrow_unchecked` guarantees the elements are
                        // valid UTF-8, and the offsets came out of a flat `PlBinaryArray`.
                        Box::new(unsafe {
                            Utf8Array::new_unchecked(dtype, offsets, values, validity)
                        })
                    }
                },
                PhysicalType::LargeUtf8 => {
                    let offsets = widen_offsets(offsets);
                    if validate_utf8 {
                        Box::new(Utf8Array::try_new(dtype, offsets, values, validity)?)
                    } else {
                        // SAFETY: as above.
                        Box::new(unsafe {
                            Utf8Array::new_unchecked(dtype, offsets, values, validity)
                        })
                    }
                },
                _ => unreachable!("the outer match bound the physical type"),
            }
        },

        PhysicalType::BinaryView | PhysicalType::Utf8View => {
            let (views, buffers, validity) = downcast_pl::<PlBinaryViewArray>(array, &dtype)?
                .to_flat()
                .into_inner();

            if physical == PhysicalType::Utf8View && validate_utf8 {
                Box::new(Utf8ViewArray::try_new(dtype, views, buffers, validity)?)
            } else if physical == PhysicalType::Utf8View {
                // SAFETY: the caller of `to_arrow_unchecked` guarantees the elements are valid
                // UTF-8, and the views came out of a flat `PlBinaryViewArray`, which validates
                // them against its buffers.
                Box::new(unsafe {
                    Utf8ViewArray::new_unchecked_unknown_md(dtype, views, buffers, validity, None)
                })
            } else {
                // SAFETY: the views came out of a `PlBinaryViewArray`, which validates them
                // against its buffers.
                Box::new(unsafe {
                    BinaryViewArray::new_unchecked_unknown_md(dtype, views, buffers, validity, None)
                })
            }
        },

        PhysicalType::FixedSizeBinary => {
            let array = downcast_pl::<PlFixedSizeBinaryArray>(array, &dtype)?;
            let ArrowDataType::FixedSizeBinary(size) = *dtype.to_storage() else {
                unreachable!("the physical type is that of a fixed size binary data type")
            };
            polars_ensure!(
                array.width() == size,
                ComputeError:
                "cannot export an array of {}-byte elements as {dtype:?}",
                array.width(),
            );

            let flat = array.to_flat();
            Box::new(FixedSizeBinaryArray::new(
                dtype,
                flat.values().clone(),
                flat.validity().cloned(),
            ))
        },

        PhysicalType::List | PhysicalType::LargeList => {
            let array = downcast_pl::<PlListArray>(array, &dtype)?;
            let (ArrowDataType::List(field) | ArrowDataType::LargeList(field)) = dtype.to_storage()
            else {
                unreachable!("the physical type is that of a list data type")
            };
            let inner = field.dtype().clone();

            let (values, offsets, validity) = array.to_flat().into_inner();
            let values = to_arrow_impl(&*values, inner, validate_utf8)?;

            match physical {
                PhysicalType::List => Box::new(ListArray::new(
                    dtype,
                    narrow_offsets(&offsets)?,
                    values,
                    validity,
                )),
                _ => Box::new(ListArray::new(
                    dtype,
                    widen_offsets(offsets),
                    values,
                    validity,
                )),
            }
        },

        PhysicalType::FixedSizeList => {
            let array = downcast_pl::<PlFixedSizeListArray>(array, &dtype)?;
            let ArrowDataType::FixedSizeList(field, size) = dtype.to_storage() else {
                unreachable!("the physical type is that of a fixed size list data type")
            };
            let (inner, size) = (field.dtype().clone(), *size);
            polars_ensure!(
                array.width() == size,
                ComputeError:
                "cannot export an array of {}-element elements as {dtype:?}",
                array.width(),
            );

            let (values, _, validity) = array.to_flat().into_inner();
            let values = to_arrow_impl(&*values, inner, validate_utf8)?;

            Box::new(FixedSizeListArray::new(dtype, length, values, validity))
        },

        PhysicalType::Struct => {
            let array = downcast_pl::<PlStructArray>(array, &dtype)?;
            let ArrowDataType::Struct(fields) = dtype.to_storage() else {
                unreachable!("the physical type is that of a struct data type")
            };
            polars_ensure!(
                array.num_fields() == fields.len(),
                ComputeError:
                "cannot export an array of {} fields as {dtype:?}, which has {}",
                array.num_fields(), fields.len(),
            );

            let values = array
                .fields()
                .iter()
                .zip(fields)
                .map(|(array, field)| to_arrow_impl(&**array, field.dtype().clone(), validate_utf8))
                .collect::<PolarsResult<Vec<_>>>()?;
            let validity = array.validity().map(|validity| validity.to_flat());

            Box::new(StructArray::new(dtype, length, values, validity))
        },

        physical @ (PhysicalType::Dictionary(_) | PhysicalType::Union | PhysicalType::Map) => {
            polars_bail!(
                ComputeError:
                "cannot export an array of polars-array as {dtype:?}: no array holds the elements \
                 of physical type {physical:?}",
            )
        },
    })
}

/// Downcasts an Arrow array whose physical type has already been matched on.
///
/// # Panics
/// Panics if `array` is not an `A`, which the physical type of its data type rules out.
#[inline]
fn downcast_arrow<A: Array + 'static>(array: &dyn Array) -> &A {
    array
        .as_any()
        .downcast_ref()
        .expect("the physical type of an arrow array determines the array it downcasts to")
}

/// Downcasts an array of this crate to the one the physical type of `dtype` asks for.
///
/// # Errors
/// This function errors if `array` is not an `A`, which is what makes `dtype` a data type of a
/// physical representation it is not in.
#[inline]
fn downcast_pl<'a, A: PlArray>(
    array: &'a dyn PlArray,
    dtype: &ArrowDataType,
) -> PolarsResult<&'a A> {
    match array.as_any().downcast_ref() {
        Some(array) => Ok(array),
        None => polars_bail!(
            ComputeError:
            "cannot export a {:?} array as {dtype:?}, which is of another physical representation",
            array.array_type(),
        ),
    }
}

/// The 64-bit counterpart of Arrow offsets, which is what the arrays of this crate hold.
///
/// This is `O(1)` for the 64-bit offsets, which have the same layout, and `O(len)` for the 32-bit
/// ones, which are widened.
fn offsets_from_arrow<O: arrow::types::Offset>(offsets: &OffsetsBuffer<O>) -> Buffer<u64> {
    if O::IS_LARGE
        && let Some(offsets) = (offsets.buffer() as &dyn std::any::Any).downcast_ref::<Buffer<i64>>()
        // SAFETY-adjacent: `i64` and `u64` have the same size and alignment, so this never fails
        // and never reinterprets a buffer of another width.
        && let Ok(offsets) = offsets.clone().try_transmute::<u64>()
    {
        return offsets;
    }

    Buffer::from(
        offsets
            .buffer()
            .iter()
            .map(|offset| offset.to_usize() as u64)
            .collect::<Vec<_>>(),
    )
}

/// The 64-bit Arrow counterpart of the offsets of this crate, which is `O(1)`.
///
/// # Panics
/// Panics if an offset exceeds [`i64::MAX`], which no offset into a buffer that fits in memory
/// does.
fn widen_offsets(offsets: Buffer<u64>) -> OffsetsBuffer<i64> {
    debug_assert!(offsets.last().is_none_or(|&last| last <= i64::MAX as u64));

    let offsets = offsets.try_transmute::<i64>().unwrap_or_else(|offsets| {
        Buffer::from(offsets.iter().map(|&o| o as i64).collect::<Vec<_>>())
    });

    // SAFETY: the offsets came out of a flat array of this crate, so they are ordered and
    // non-negative, and widening preserves both.
    unsafe { OffsetsBuffer::new_unchecked(offsets) }
}

/// The 32-bit Arrow counterpart of the offsets of this crate, which is `O(len)`.
///
/// # Errors
/// This function errors if an offset does not fit in an [`i32`].
fn narrow_offsets(offsets: &Buffer<u64>) -> PolarsResult<OffsetsBuffer<i32>> {
    let last = offsets.last().copied().unwrap_or(0);
    polars_ensure!(
        last <= i32::MAX as u64,
        ComputeError:
        "cannot export {last} bytes of elements behind 32-bit offsets",
    );

    let offsets = Buffer::from(offsets.iter().map(|&o| o as i32).collect::<Vec<_>>());

    // SAFETY: the offsets came out of a flat array of this crate, so they are ordered and
    // non-negative, and every one of them was just checked to fit in an `i32`.
    Ok(unsafe { OffsetsBuffer::new_unchecked(offsets) })
}

/// Imports the buffers of an Arrow binary or string array as a [`PlBinaryArray`].
fn binary_from_arrow<O: arrow::types::Offset>(
    values: Buffer<u8>,
    offsets: &OffsetsBuffer<O>,
    length: usize,
    validity: Option<Bitmap>,
) -> Box<dyn PlArray> {
    // SAFETY: the offsets of an Arrow array are ordered, hold one per element plus the end of the
    // last, and end within the values; its validity mask holds one bit per element.
    Box::new(unsafe {
        PlBinaryArray::new_unchecked(values, offsets_from_arrow(offsets), length, validity)
    })
}

/// Imports the buffers of an Arrow binary view or string view array as a [`PlBinaryViewArray`].
fn view_from_arrow(
    views: Buffer<View>,
    buffers: Buffer<Buffer<u8>>,
    length: usize,
    validity: Option<Bitmap>,
) -> Box<dyn PlArray> {
    // SAFETY: the views of an Arrow view array read bytes its buffers hold, and there is one per
    // element, as there is one validity bit per element.
    Box::new(unsafe { PlBinaryViewArray::new_unchecked(views, buffers, length, validity) })
}

/// Imports the values and offsets of an Arrow list array as a [`PlListArray`].
fn list_from_arrow<O: arrow::types::Offset>(
    values: &dyn Array,
    offsets: &OffsetsBuffer<O>,
    length: usize,
    validity: Option<Bitmap>,
) -> PolarsResult<Box<dyn PlArray>> {
    let values = from_arrow(values)?;

    // SAFETY: the offsets of an Arrow array are ordered, hold one per element plus the end of the
    // last, and end within the values; its validity mask holds one bit per element.
    Ok(Box::new(unsafe {
        PlListArray::new_unchecked(values, offsets_from_arrow(offsets), length, validity)
    }))
}

/// Runs `$body` with `$T` bound to the Rust type of the Arrow [`PrimitiveType`] `$primitive`.
///
/// This evaluates to `Some(body)` for the element types that are a [`NativeType`], and to `None`
/// for [`PrimitiveType::MonthDayMillis`], which is not one and therefore is not the element type of
/// any array.
macro_rules! with_primitive_type {
    ($primitive:expr, |$T:ident| $body:expr $(,)?) => {{
        macro_rules! run {
            ($element:ty) => {{
                #[allow(dead_code)]
                type $T = $element;
                Some($body)
            }};
        }

        match $primitive {
            PrimitiveType::Int8 => run!(i8),
            PrimitiveType::Int16 => run!(i16),
            PrimitiveType::Int32 => run!(i32),
            PrimitiveType::Int64 => run!(i64),
            PrimitiveType::Int128 => run!(i128),
            PrimitiveType::Int256 => run!(i256),
            PrimitiveType::UInt8 => run!(u8),
            PrimitiveType::UInt16 => run!(u16),
            PrimitiveType::UInt32 => run!(u32),
            PrimitiveType::UInt64 => run!(u64),
            PrimitiveType::UInt128 => run!(u128),
            PrimitiveType::Float16 => run!(pf16),
            PrimitiveType::Float32 => run!(f32),
            PrimitiveType::Float64 => run!(f64),
            PrimitiveType::DaysMs => run!(days_ms),
            PrimitiveType::MonthDayNano => run!(months_days_ns),
            PrimitiveType::MonthDayMillis => None,
        }
    }};
}

/// Imports an Arrow primitive array of `primitive` as a [`PlPrimitiveArray`].
fn primitive_from_arrow(
    array: &dyn Array,
    primitive: PrimitiveType,
    length: usize,
    validity: Option<Bitmap>,
) -> PolarsResult<Box<dyn PlArray>> {
    let imported = with_primitive_type!(primitive, |T| {
        let array = downcast_arrow::<PrimitiveArray<T>>(array);
        // SAFETY: the values of an Arrow primitive array hold one slot per element, as does its
        // validity mask, which is what makes them flat here.
        Box::new(unsafe {
            PlPrimitiveArray::<T>::new_unchecked(array.values().clone(), length, validity)
        }) as Box<dyn PlArray>
    });

    match imported {
        Some(array) => Ok(array),
        None => polars_bail!(
            ComputeError:
            "cannot import an arrow array of elements of primitive type {primitive:?}: they are \
             of no rust type",
        ),
    }
}

/// Exports a [`PlPrimitiveArray`] as an Arrow primitive array of `dtype`.
fn primitive_to_arrow(
    array: &dyn PlArray,
    primitive: PrimitiveType,
    dtype: &ArrowDataType,
) -> PolarsResult<Box<dyn Array>> {
    // The element type is taken from the array rather than from `primitive`, which does not pin it
    // down: a `View` and a `u128` are both `PrimitiveType::UInt128`.
    let exported = with_match_pl_primitive_array_type!(array, |T| {
        polars_ensure!(
            <T as NativeType>::PRIMITIVE == primitive,
            ComputeError:
            "cannot export an array of {:?} elements as {dtype:?}",
            <T as NativeType>::PRIMITIVE,
        );

        let array = array
            .as_any()
            .downcast_ref::<PlPrimitiveArray<T>>()
            .unwrap();
        let (values, validity) = array.to_flat().into_inner();
        Ok(Box::new(PrimitiveArray::new(dtype.clone(), values, validity)) as Box<dyn Array>)
    });

    match exported {
        Some(array) => array,
        None => polars_bail!(
            ComputeError:
            "cannot export a {:?} array as {dtype:?}, which is of another physical representation",
            array.array_type(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{DictionaryArray, Int32Array, Int64Array, UInt32Array};
    use arrow::datatypes::{Field, IntegerType};
    use polars_utils::pl_str::PlSmallStr;

    use super::*;
    use crate::PlArrayType;

    fn field(dtype: ArrowDataType) -> Box<Field> {
        Box::new(Field::new(PlSmallStr::from_static("item"), dtype, true))
    }

    /// Every Arrow array that has a counterpart in this crate, next to the data type it is of.
    ///
    /// The data types are the logical ones wherever they differ from the physical representation,
    /// so that a round trip through this crate is shown to keep them.
    fn arrays() -> Vec<Box<dyn Array>> {
        let values = Int32Array::from_slice([1, 2, 3, 4]).boxed();

        vec![
            NullArray::new(ArrowDataType::Null, 3).boxed(),
            BooleanArray::from([Some(true), None, Some(false)]).boxed(),
            Int32Array::from([Some(1), None, Some(3)])
                .to(ArrowDataType::Date32)
                .boxed(),
            Int64Array::from_slice([1, 2, 3])
                .to(ArrowDataType::Timestamp(
                    arrow::datatypes::TimeUnit::Microsecond,
                    None,
                ))
                .boxed(),
            BinaryArray::<i64>::from([Some(b"foo".as_slice()), None, Some(b"")]).boxed(),
            BinaryArray::<i32>::from([Some(b"foo".as_slice()), None, Some(b"")]).boxed(),
            Utf8Array::<i64>::from([Some("foo"), None, Some("")]).boxed(),
            Utf8Array::<i32>::from([Some("foo"), None, Some("")]).boxed(),
            BinaryViewArray::from_slice([Some(b"foo".as_slice()), None, Some(b"a long value")])
                .boxed(),
            Utf8ViewArray::from_slice([Some("foo"), None, Some("a value that is not inlined")])
                .boxed(),
            FixedSizeBinaryArray::from([Some([1u8, 2]), None, Some([5, 6])]).boxed(),
            ListArray::<i64>::new(
                ArrowDataType::LargeList(field(ArrowDataType::Int32)),
                OffsetsBuffer::try_from(Buffer::from(vec![0i64, 2, 2, 4])).unwrap(),
                values.clone(),
                Some(Bitmap::from_iter([true, false, true])),
            )
            .boxed(),
            ListArray::<i32>::new(
                ArrowDataType::List(field(ArrowDataType::Int32)),
                OffsetsBuffer::try_from(Buffer::from(vec![0i32, 2, 2, 4])).unwrap(),
                values.clone(),
                None,
            )
            .boxed(),
            FixedSizeListArray::new(
                ArrowDataType::FixedSizeList(field(ArrowDataType::Int32), 2),
                2,
                values.clone(),
                Some(Bitmap::from_iter([true, false])),
            )
            .boxed(),
            StructArray::new(
                ArrowDataType::Struct(vec![*field(ArrowDataType::Int32)]),
                4,
                vec![values.clone()],
                Some(Bitmap::from_iter([true, false, true, true])),
            )
            .boxed(),
            // A struct of no fields is nothing but a length and a mask.
            StructArray::new(ArrowDataType::Struct(vec![]), 3, vec![], None).boxed(),
        ]
    }

    #[test]
    fn every_arrow_array_round_trips() {
        for arrow in arrays() {
            let array = from_arrow(&*arrow).unwrap();
            assert_eq!(array.len(), arrow.len());
            assert_eq!(array.null_count(), arrow.null_count());

            let exported = to_arrow(&*array, arrow.dtype().clone()).unwrap();
            assert_eq!(exported.dtype(), arrow.dtype());
            assert_eq!(&exported, &arrow, "{:?} did not round trip", arrow.dtype());
        }
    }

    #[test]
    fn every_sliced_arrow_array_round_trips() {
        for arrow in arrays() {
            let arrow = arrow.sliced(1, arrow.len() - 1);

            let array = from_arrow(&*arrow).unwrap();
            assert_eq!(array.len(), arrow.len());

            let exported = to_arrow(&*array, arrow.dtype().clone()).unwrap();
            assert_eq!(&exported, &arrow, "{:?} did not round trip", arrow.dtype());
        }
    }

    #[test]
    fn importing_drops_the_logical_type() {
        // The physical representation is all that is left: a date is the integer under it, and a
        // string the bytes under it.
        let arrow = Int32Array::from_slice([1]).to(ArrowDataType::Date32);
        assert_eq!(
            from_arrow(&arrow).unwrap().array_type(),
            PlArrayType::Primitive(PrimitiveType::Int32),
        );

        let arrow = Utf8ViewArray::from_slice_values(["foo"]);
        let array = from_arrow(&arrow).unwrap();
        assert_eq!(array.array_type(), PlArrayType::BinaryView);
        assert_eq!(
            array
                .as_any()
                .downcast_ref::<PlBinaryViewArray>()
                .unwrap()
                .value(0),
            b"foo",
        );

        // Which is why exporting takes the data type to export as, and takes any data type of that
        // representation.
        let exported = to_arrow(&*array, ArrowDataType::BinaryView).unwrap();
        assert_eq!(exported.dtype(), &ArrowDataType::BinaryView);
    }

    #[test]
    fn importing_shares_the_buffers() {
        let arrow = Int32Array::from_slice([1, 2, 3]);
        let array = from_arrow(&arrow).unwrap();
        let array = array
            .as_any()
            .downcast_ref::<PlPrimitiveArray<i32>>()
            .unwrap();

        assert!(array.flat_values().unwrap().is_same_buffer(arrow.values()),);

        // And so does exporting, for an array that is already flat.
        let exported = to_arrow(array, ArrowDataType::Int32).unwrap();
        let exported = exported.as_any().downcast_ref::<Int32Array>().unwrap();
        assert!(exported.values().is_same_buffer(arrow.values()));
    }

    #[test]
    fn sixty_four_bit_offsets_are_shared_and_thirty_two_bit_ones_widened() {
        let arrow = BinaryArray::<i64>::from_slice([b"foo", b"bar"]);
        let array = from_arrow(&arrow).unwrap();
        let array = array.as_any().downcast_ref::<PlBinaryArray>().unwrap();
        assert_eq!(array.flat_offsets().unwrap().as_slice(), [0, 3, 6]);
        assert_eq!(
            array.flat_offsets().unwrap().storage_ptr().cast::<i64>(),
            arrow.offsets().buffer().storage_ptr(),
        );

        let arrow = BinaryArray::<i32>::from_slice([b"foo", b"bar"]);
        let array = from_arrow(&arrow).unwrap();
        let array = array.as_any().downcast_ref::<PlBinaryArray>().unwrap();
        assert_eq!(array.flat_offsets().unwrap().as_slice(), [0, 3, 6]);
    }

    #[test]
    fn exporting_writes_out_a_scalar_array() {
        // Arrow has no scalar representation, so the shared value is written out once per element.
        let array = PlPrimitiveArray::new_scalar(7i32, 3);
        assert!(array.flat_values().is_none());

        let exported = to_arrow(&array, ArrowDataType::Int32).unwrap();
        assert_eq!(
            exported
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .as_slice(),
            [7, 7, 7],
        );

        // Including the one behind a scalar validity mask.
        let array = PlBinaryViewArray::new_full_null(2);
        let exported = to_arrow(&array, ArrowDataType::BinaryView).unwrap();
        assert_eq!(exported.null_count(), 2);
        assert_eq!(exported.validity().unwrap().len(), 2);

        // And the lists of a scalar list array, which are laid end to end.
        let array = PlListArray::new_scalar(Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])), 3);
        let exported = to_arrow(
            &array,
            ArrowDataType::LargeList(field(ArrowDataType::Int32)),
        )
        .unwrap();
        let exported = exported.as_any().downcast_ref::<ListArray<i64>>().unwrap();
        assert_eq!(exported.offsets().buffer().as_slice(), [0, 2, 4, 6]);
        assert_eq!(exported.values().len(), 6);
    }

    #[test]
    fn exporting_as_a_string_type_validates_utf8() {
        let array = PlBinaryViewArray::from_values_iter([b"\xff".as_slice()]);
        assert!(to_arrow(&array, ArrowDataType::Utf8View).is_err());
        assert!(to_arrow(&array, ArrowDataType::BinaryView).is_ok());

        // The unchecked counterpart is what a caller that already knows uses.
        let valid = PlBinaryViewArray::from_values_iter([b"foo".as_slice()]);
        let exported = unsafe { to_arrow_unchecked(&valid, ArrowDataType::Utf8View) }.unwrap();
        assert_eq!(
            exported
                .as_any()
                .downcast_ref::<Utf8ViewArray>()
                .unwrap()
                .value(0),
            "foo",
        );

        let array = PlBinaryArray::from_values_iter([b"\xff".as_slice()]);
        assert!(to_arrow(&array, ArrowDataType::LargeUtf8).is_err());
        assert!(to_arrow(&array, ArrowDataType::Utf8).is_err());
        assert!(to_arrow(&array, ArrowDataType::LargeBinary).is_ok());
    }

    #[test]
    fn exporting_as_another_representation_is_rejected() {
        let array = PlPrimitiveArray::from_vec(vec![1i32]);
        assert!(to_arrow(&array, ArrowDataType::Int64).is_err());
        assert!(to_arrow(&array, ArrowDataType::Boolean).is_err());
        assert!(to_arrow(&array, ArrowDataType::Utf8View).is_err());

        let array = PlBooleanArray::from_vec(vec![true]);
        assert!(to_arrow(&array, ArrowDataType::Int32).is_err());

        // A width is part of the representation.
        let array = PlFixedSizeBinaryArray::from_vec(vec![1u8, 2], 2);
        assert!(to_arrow(&array, ArrowDataType::FixedSizeBinary(2)).is_ok());
        assert!(to_arrow(&array, ArrowDataType::FixedSizeBinary(1)).is_err());

        let array = PlFixedSizeListArray::from_values(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            2,
        );
        assert!(
            to_arrow(
                &array,
                ArrowDataType::FixedSizeList(field(ArrowDataType::Int32), 1),
            )
            .is_err()
        );

        // As is the number of fields of a struct.
        let array =
            PlStructArray::from_fields(vec![Box::new(PlPrimitiveArray::from_vec(vec![1i32]))]);
        assert!(to_arrow(&array, ArrowDataType::Struct(vec![])).is_err());
    }

    #[test]
    fn arrays_with_no_counterpart_are_rejected() {
        let arrow = DictionaryArray::try_new(
            ArrowDataType::Dictionary(
                IntegerType::UInt32,
                Box::new(ArrowDataType::Utf8View),
                false,
            ),
            UInt32Array::from_slice([0, 0]),
            Utf8ViewArray::from_slice_values(["foo"]).boxed(),
        )
        .unwrap();
        assert!(from_arrow(&arrow).is_err());

        // And so is exporting as one: nothing here holds dictionary-encoded elements.
        let array = PlBinaryViewArray::from_values_iter([b"foo".as_slice()]);
        assert!(to_arrow(&array, arrow.dtype().clone()).is_err());
    }

    #[test]
    fn nesting_round_trips() {
        // A large list of structs of a string view and a fixed size list of integers.
        let inner = StructArray::new(
            ArrowDataType::Struct(vec![
                Field::new(PlSmallStr::from_static("s"), ArrowDataType::Utf8View, true),
                Field::new(
                    PlSmallStr::from_static("a"),
                    ArrowDataType::FixedSizeList(field(ArrowDataType::Int32), 2),
                    true,
                ),
            ]),
            2,
            vec![
                Utf8ViewArray::from_slice([Some("foo"), None]).boxed(),
                FixedSizeListArray::new(
                    ArrowDataType::FixedSizeList(field(ArrowDataType::Int32), 2),
                    2,
                    Int32Array::from_slice([1, 2, 3, 4]).boxed(),
                    None,
                )
                .boxed(),
            ],
            None,
        );
        let dtype = ArrowDataType::LargeList(field(inner.dtype().clone()));
        let arrow = ListArray::<i64>::new(
            dtype.clone(),
            OffsetsBuffer::try_from(Buffer::from(vec![0i64, 1, 2])).unwrap(),
            inner.boxed(),
            None,
        );

        let array = from_arrow(&arrow).unwrap();
        assert_eq!(array.array_type(), PlArrayType::List);

        let exported = to_arrow(&*array, dtype).unwrap();
        assert_eq!(&exported, &(arrow.boxed()));
    }

    #[test]
    fn an_extension_type_is_the_representation_it_wraps() {
        let inner = ArrowDataType::Int32;
        let dtype = ArrowDataType::Extension(Box::new(arrow::datatypes::ExtensionType {
            name: PlSmallStr::from_static("ext"),
            inner,
            metadata: None,
        }));

        let arrow = Int32Array::from_slice([1, 2]).to(dtype.clone());
        let array = from_arrow(&arrow).unwrap();
        assert_eq!(
            array.array_type(),
            PlArrayType::Primitive(PrimitiveType::Int32),
        );

        let exported = to_arrow(&*array, dtype.clone()).unwrap();
        assert_eq!(exported.dtype(), &dtype);
    }

    #[test]
    fn an_empty_array_round_trips() {
        for arrow in arrays() {
            let arrow = arrow.sliced(0, 0);
            let array = from_arrow(&*arrow).unwrap();
            assert!(array.is_empty());
            assert_eq!(&to_arrow(&*array, arrow.dtype().clone()).unwrap(), &arrow);
        }
    }

    #[test]
    fn every_primitive_element_type_round_trips() {
        /// An empty Arrow array of `T` elements typed as `dtype`, imported and exported again.
        fn round_trip<T: NativeType>(dtype: ArrowDataType) {
            let arrow = PrimitiveArray::<T>::new_empty(dtype.clone());
            let array = from_arrow(&arrow).unwrap();
            assert_eq!(
                array.array_type(),
                PlArrayType::Primitive(<T as NativeType>::PRIMITIVE),
            );

            // The concrete arrays are compared rather than the trait objects, whose `PartialEq`
            // does not reach every element type.
            let exported = to_arrow(&*array, dtype).unwrap();
            assert_eq!(
                exported
                    .as_any()
                    .downcast_ref::<PrimitiveArray<T>>()
                    .unwrap(),
                &arrow,
            );
        }

        use arrow::datatypes::IntervalUnit;
        round_trip::<i8>(ArrowDataType::Int8);
        round_trip::<i16>(ArrowDataType::Int16);
        round_trip::<i32>(ArrowDataType::Int32);
        round_trip::<i64>(ArrowDataType::Int64);
        round_trip::<i128>(ArrowDataType::Int128);
        round_trip::<i256>(ArrowDataType::Decimal256(10, 2));
        round_trip::<u8>(ArrowDataType::UInt8);
        round_trip::<u16>(ArrowDataType::UInt16);
        round_trip::<u32>(ArrowDataType::UInt32);
        round_trip::<u64>(ArrowDataType::UInt64);
        round_trip::<u128>(ArrowDataType::UInt128);
        round_trip::<pf16>(ArrowDataType::Float16);
        round_trip::<f32>(ArrowDataType::Float32);
        round_trip::<f64>(ArrowDataType::Float64);
        round_trip::<days_ms>(ArrowDataType::Interval(IntervalUnit::DayTime));
        round_trip::<months_days_ns>(ArrowDataType::Interval(IntervalUnit::MonthDayNano));

        // `PrimitiveType::MonthDayMillis` is the one primitive type of no rust type, so there is
        // no arrow array of it to import: `from_arrow` reports it rather than panicking, but
        // nothing can be handed to it that reaches that arm.
    }

    #[test]
    fn a_null_array_is_a_representation_like_any_other() {
        let array = PlNullArray::new(3);
        assert!(to_arrow(&array, ArrowDataType::Null).is_ok());
        assert!(to_arrow(&array, ArrowDataType::Int32).is_err());

        // Nothing but a null array exports as `Null`: every other one holds values it would drop.
        let array = PlPrimitiveArray::<i32>::new_full_null(3);
        assert!(to_arrow(&array, ArrowDataType::Null).is_err());
    }
}
