//! Casting the arrays of `polars-array`, on a pair of Polars types or of Arrow ones.

mod arrow_kernels;
mod native;
pub mod temporal;

use arrow::array::Array;
use arrow::datatypes::{ArrowDataType, TimeUnit as ArrowTimeUnit};
pub use arrow_kernels::*;
pub use native::*;
use polars_array::PlArray;
use polars_array::arrow::import;
use polars_dtype::DataType;
use polars_error::{PolarsResult, polars_bail};

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

/// The Polars type whose values `array` holds, which is the physical type of a logical one: an
/// array carries no name over its values, so a cast off one reads the buffers it is laid out in.
pub fn physical_dtype(array: &dyn PlArray) -> DataType {
    use polars_array::PlArrayType as A;

    match array.array_type() {
        A::Null => DataType::Null,
        A::Boolean => DataType::Boolean,
        A::Primitive(primitive) => match primitive {
            polars_array::PrimitiveType::Int8 => DataType::Int8,
            polars_array::PrimitiveType::Int16 => DataType::Int16,
            polars_array::PrimitiveType::Int32 => DataType::Int32,
            polars_array::PrimitiveType::Int64 => DataType::Int64,
            polars_array::PrimitiveType::Int128 => DataType::Int128,
            polars_array::PrimitiveType::UInt8 => DataType::UInt8,
            polars_array::PrimitiveType::UInt16 => DataType::UInt16,
            polars_array::PrimitiveType::UInt32 => DataType::UInt32,
            polars_array::PrimitiveType::UInt64 => DataType::UInt64,
            polars_array::PrimitiveType::UInt128 => DataType::UInt128,
            polars_array::PrimitiveType::Float16 => DataType::Float16,
            polars_array::PrimitiveType::Float32 => DataType::Float32,
            polars_array::PrimitiveType::Float64 => DataType::Float64,
            primitive => unimplemented!("polars-compute: {primitive:?} is no type of Polars"),
        },
        A::Binary => DataType::BinaryOffset,
        A::BinaryView => DataType::Binary,
        A::Utf8View => DataType::String,
        A::List => DataType::List(Box::new(physical_dtype(
            downcast::<polars_array::PlListArray>(array).values(),
        ))),
        #[cfg(feature = "dtype-array")]
        A::FixedSizeList => {
            let array = downcast::<polars_array::PlFixedSizeListArray>(array);
            DataType::Array(Box::new(physical_dtype(array.values())), array.width())
        },
        #[cfg(feature = "dtype-struct")]
        A::Struct => DataType::Struct(
            downcast::<polars_array::PlStructArray>(array)
                .fields()
                .iter()
                .enumerate()
                .map(|(i, field)| {
                    polars_dtype::Field::new(
                        polars_utils::format_pl_smallstr!("{i}"),
                        physical_dtype(&**field),
                    )
                })
                .collect(),
        ),
        array_type => {
            unimplemented!("polars-compute: {array_type:?} has no type of Polars to cast on")
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

/// Casts an Arrow array to `to_type`, over the arrays of `polars-array` that hold its values.
///
/// The array crosses over to `polars-array` first, which shares its buffers rather than reading
/// them — except for the 32-bit offsets that no array there holds, which are widened.
///
/// # Panics
/// A dictionary, a map or a union *nested inside* `array` panics in the crossing over rather than
/// erroring here: reading the pair as one that has no answer would mean walking the type first,
/// which is what [`cast_crossed_over`] does not do. Nothing casts such a column — the nested Arrow
/// arrays arrive through `polars_core::series::from`, which walks them itself.
pub fn cast_arrow(
    array: &dyn Array,
    to_type: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn PlArray>> {
    let from_type = array.dtype();

    // A dictionary is the one Arrow array whose elements no array of `polars-array` holds, which
    // is what the crossing over would panic on — see `polars_compute::cast::cast_to_dictionary`.
    if matches!(from_type, ArrowDataType::Dictionary(..)) {
        return unsupported(from_type, to_type);
    }

    cast_crossed_over(&*import::from_arrow(array), from_type, to_type, options)
}

/// Casts an array that crossed over holding the values of `from_type` to `to_type`.
///
/// A nested pair recurses on the Arrow types of the children rather than on a Polars type built
/// out of them: an Arrow type is read as the Polars type of the *values* it holds, which is a
/// type that needs no memory of its own — and a nested one does.
fn cast_crossed_over(
    array: &dyn PlArray,
    from_type: &ArrowDataType,
    to_type: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn PlArray>> {
    use ArrowDataType as A;

    // The values of both sides are laid out the same way and read the same way, which leaves the
    // array itself as the answer — including for a nested type, whose children the cast below
    // would otherwise walk one by one.
    if from_type == to_type {
        return Ok(array.to_boxed());
    }

    let recurse = |values: &dyn PlArray, from: &ArrowDataType, to: &ArrowDataType| {
        cast_crossed_over(values, from, to, options)
    };

    match (from_type, to_type) {
        (
            A::List(from_field) | A::LargeList(from_field),
            A::List(to_field) | A::LargeList(to_field),
        ) => {
            let out = native::cast_list(downcast(array), |values| {
                recurse(values, from_field.dtype(), to_field.dtype())
            })?;
            Ok(Box::new(out))
        },
        #[cfg(feature = "dtype-array")]
        (A::FixedSizeList(from_field, from_width), A::FixedSizeList(to_field, to_width)) => {
            polars_error::polars_ensure!(
                from_width == to_width,
                InvalidOperation: "cannot cast Array to a different width"
            );
            let out = native::cast_fixed_size_list(downcast(array), |values| {
                recurse(values, from_field.dtype(), to_field.dtype())
            })?;
            Ok(Box::new(out))
        },
        #[cfg(feature = "dtype-array")]
        (A::List(from_field) | A::LargeList(from_field), A::FixedSizeList(to_field, width)) => {
            let out = native::list_to_fixed_size_list(downcast(array), *width, |values| {
                recurse(values, from_field.dtype(), to_field.dtype())
            })?;
            Ok(Box::new(out))
        },
        #[cfg(feature = "dtype-array")]
        (A::FixedSizeList(from_field, _), A::List(to_field) | A::LargeList(to_field)) => {
            let out = native::fixed_size_list_to_list(downcast(array), |values| {
                recurse(values, from_field.dtype(), to_field.dtype())
            })?;
            Ok(Box::new(out))
        },
        #[cfg(feature = "dtype-struct")]
        (A::Struct(from_fields), A::Struct(to_fields)) => {
            polars_error::polars_ensure!(
                from_fields.len() == to_fields.len(),
                InvalidOperation: "Cannot cast struct with different number of fields."
            );
            let out = native::cast_struct(downcast(array), |i, field| {
                recurse(field, from_fields[i].dtype(), to_fields[i].dtype())
            })?;
            Ok(Box::new(out))
        },

        // The bytes of an element are held one per value, which is what makes the two readable as
        // one another.
        (A::List(field) | A::LargeList(field), A::BinaryView) if field.dtype() == &A::UInt8 => {
            Ok(Box::new(native::list_uint8_to_binview(downcast(array))?))
        },
        (A::BinaryView, A::List(field) | A::LargeList(field)) if field.dtype() == &A::UInt8 => {
            let bytes = native::view_to_binary(downcast(array));
            Ok(Box::new(native::binary_to_list(&bytes)))
        },

        // A fixed size binary array crossed over as the one array no Polars type names, and the
        // bytes it holds are read as a binary's.
        (A::FixedSizeBinary(_), _) => {
            let view = native::fixed_size_binary_to_binview(downcast(array));
            let Some(to) = datatype_of(to_type) else {
                return unsupported(from_type, to_type);
            };
            native::cast(&view, &DataType::Binary, &to, options)
        },

        _ => {
            let (Some(from), Some(to)) = (datatype_of(from_type), datatype_of(to_type)) else {
                return unsupported(from_type, to_type);
            };

            // A nested type names the values of its children rather than any of its own, so a
            // pair of them is answered by the arms above: what is left here is a pair they hold
            // no answer for.
            if from.is_nested() || to.is_nested() {
                return unsupported(from_type, to_type);
            }

            native::cast(array, &from, &to, options)
        },
    }
}

/// The Polars type whose values an Arrow array of `dtype` holds once it has crossed over, or
/// `None` for the Arrow types no Polars type names.
///
/// This is [`DataType::from_arrow_dtype`] but for the types below, which are the ones whose values
/// do not cross over laid out the way the Polars type they *mean* is laid out. It answers for the
/// nested types too, whose pairs [`cast_crossed_over`] handles itself — which is why the pair it
/// hands to a kernel is checked not to be one.
fn datatype_of(dtype: &ArrowDataType) -> Option<DataType> {
    use ArrowDataType as A;

    Some(match dtype {
        // These crossed over as the bytes they hold, with their offsets widened. A string means a
        // string, but it is held in a view array, which is not what they are laid out as.
        A::Utf8 | A::LargeUtf8 | A::Binary | A::LargeBinary => DataType::BinaryOffset,
        // A time of Polars counts nanoseconds, so a count of anything else is the count it is.
        A::Time32(_) => DataType::Int32,
        A::Time64(unit) if !matches!(unit, ArrowTimeUnit::Nanosecond) => DataType::Int64,
        // A time zone names no value of a timestamp, so which one it is stays the caller's to read
        // off the type it asked for — and rejecting one it does not know is not a cast's to do.
        A::Timestamp(unit, _) => DataType::Datetime(unit.into(), None),

        // A decimal of another width crossed over as the integer holding it, which is not the
        // `i128` a decimal of Polars is held in; an interval crossed over as its own element type,
        // which no Polars type names; and a fixed size binary is read as the bytes it holds by
        // the arm of [`cast_crossed_over`] that it reaches first, never as a target.
        A::Decimal32(..) | A::Decimal64(..) | A::Decimal256(..) => return None,
        A::Interval(_) | A::FixedSizeBinary(_) => return None,
        // A dictionary is read as the type of its values, which is no answer for a cast *to* one:
        // packing the values is `cast_to_dictionary`'s to do, and nothing here holds a dictionary.
        A::Dictionary(..) => return None,

        dtype => DataType::from_arrow_dtype(dtype),
    })
}

fn unsupported<T>(from_type: &ArrowDataType, to_type: &ArrowDataType) -> PolarsResult<T> {
    polars_bail!(InvalidOperation: "casting from {from_type:?} to {to_type:?} not supported")
}

#[cfg(test)]
mod tests {
    use arrow::array::{ListArray, PrimitiveArray, StructArray, Utf8Array};
    use arrow::datatypes::Field;
    use arrow::offset::OffsetsBuffer;
    use polars_array::{PlListArray, PlPrimitiveArray, PlStructArray, PlUtf8ViewArray};

    use super::*;

    fn large_list_of(values: Box<dyn Array>, offsets: Vec<i64>) -> ListArray<i64> {
        let dtype = ListArray::<i64>::default_datatype(values.dtype().clone());
        let offsets = OffsetsBuffer::try_from(offsets).unwrap();
        ListArray::<i64>::new(dtype, offsets, values, None)
    }

    /// The pair a nested cast is dispatched on is the pair of the children, which are the arrays
    /// the values are held in.
    #[test]
    fn cast_arrow_casts_the_values_of_a_list() {
        let array = large_list_of(
            PrimitiveArray::from_slice([1i32, 2, 3]).boxed(),
            vec![0, 2, 3],
        );
        let to_type = ListArray::<i64>::default_datatype(ArrowDataType::Int64);

        let out = cast_arrow(&array, &to_type, CastOptionsImpl::default()).unwrap();

        let out: &PlListArray = downcast(&*out);
        assert_eq!(out.len(), 2);
        assert_eq!(out.value_range(0), 0..2);
        let values: &PlPrimitiveArray<i64> = out.values().as_any().downcast_ref().unwrap();
        assert_eq!(
            (values.value(0), values.value(1), values.value(2)),
            (1, 2, 3)
        );
    }

    /// A cast that changes nothing reads the same values, which a nested type answers for without
    /// walking its children.
    #[test]
    fn cast_arrow_answers_a_nested_identity_with_the_array() {
        let array = large_list_of(
            PrimitiveArray::from_slice([1i32, 2, 3]).boxed(),
            vec![0, 2, 3],
        );
        let to_type = array.dtype().clone();

        let out = cast_arrow(&array, &to_type, CastOptionsImpl::default()).unwrap();

        let out: &PlListArray = downcast(&*out);
        let values: &PlPrimitiveArray<i32> = out.values().as_any().downcast_ref().unwrap();
        assert_eq!(values.len(), 3);
    }

    /// Every field of a struct is cast on the pair its own two Arrow types name.
    #[test]
    fn cast_arrow_casts_every_field_of_a_struct() {
        let fields = vec![
            Field::new("a".into(), ArrowDataType::Int32, true),
            Field::new("b".into(), ArrowDataType::LargeUtf8, true),
        ];
        let array = StructArray::new(
            ArrowDataType::Struct(fields),
            2,
            vec![
                PrimitiveArray::from_slice([7i32, 8]).boxed(),
                Utf8Array::<i64>::from_slice(["one", "two"]).boxed(),
            ],
            None,
        );
        let to_type = ArrowDataType::Struct(vec![
            Field::new("a".into(), ArrowDataType::Int64, true),
            Field::new("b".into(), ArrowDataType::Utf8View, true),
        ]);

        let out = cast_arrow(&array, &to_type, CastOptionsImpl::default()).unwrap();

        let out: &PlStructArray = downcast(&*out);
        let a: &PlPrimitiveArray<i64> = out.fields()[0].as_any().downcast_ref().unwrap();
        let b: &PlUtf8ViewArray = out.fields()[1].as_any().downcast_ref().unwrap();
        assert_eq!((a.value(0), a.value(1)), (7, 8));
        assert_eq!((b.value(0), b.value(1)), ("one", "two"));
    }

    /// A nested type is read as a type of its own, which is no answer for a pair the arms that
    /// walk the children do not hold.
    #[test]
    fn cast_arrow_reads_no_nested_type_off_a_flat_one() {
        let array = PrimitiveArray::from_slice([1i32, 2, 3]);
        let to_type = ListArray::<i64>::default_datatype(ArrowDataType::Int32);

        assert!(cast_arrow(&array, &to_type, CastOptionsImpl::default()).is_err());
    }

    /// A dictionary is the one Arrow array whose elements no array of `polars-array` holds.
    #[test]
    fn cast_arrow_reads_no_dictionary() {
        let array = PrimitiveArray::from_slice([1i32, 2, 3]);
        let to_type = ArrowDataType::Dictionary(
            arrow::datatypes::IntegerType::UInt32,
            Box::new(ArrowDataType::Int32),
            false,
        );

        assert!(cast_arrow(&array, &to_type, CastOptionsImpl::default()).is_err());
    }
}
