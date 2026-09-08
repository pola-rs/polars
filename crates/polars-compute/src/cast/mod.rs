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
pub fn cast_arrow(
    array: &dyn Array,
    to_type: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn PlArray>> {
    let from_type = array.dtype();

    // A fixed size binary array is the one array that crosses over as no type of Polars, and the
    // bytes it holds are read as a binary's.
    if let ArrowDataType::FixedSizeBinary(_) = from_type {
        let array = import::fixed_size_binary_from_arrow(
            array
                .as_any()
                .downcast_ref()
                .expect("the data type names the array holding its values"),
        );
        let array = native::fixed_size_binary_to_binview(&array);
        let Some(to) = datatype_of(to_type) else {
            return unsupported(from_type, to_type);
        };
        return native::cast(&array, &DataType::Binary, &to, options);
    }

    let (Some(from), Some(to)) = (datatype_of(from_type), datatype_of(to_type)) else {
        return unsupported(from_type, to_type);
    };

    native::cast(&*import::from_arrow(array), &from, &to, options)
}

/// The Polars type whose values an Arrow array of `dtype` holds once it has crossed over, or
/// `None` for the Arrow types no Polars type names.
fn datatype_of(dtype: &ArrowDataType) -> Option<DataType> {
    use ArrowDataType as A;

    Some(match dtype {
        A::Null => DataType::Null,
        A::Boolean => DataType::Boolean,
        A::UInt8 => DataType::UInt8,
        A::UInt16 => DataType::UInt16,
        A::UInt32 => DataType::UInt32,
        A::UInt64 => DataType::UInt64,
        A::UInt128 => DataType::UInt128,
        A::Int8 => DataType::Int8,
        A::Int16 => DataType::Int16,
        A::Int32 => DataType::Int32,
        A::Int64 => DataType::Int64,
        A::Int128 => DataType::Int128,
        A::Float16 => DataType::Float16,
        A::Float32 => DataType::Float32,
        A::Float64 => DataType::Float64,
        #[cfg(feature = "dtype-decimal")]
        A::Decimal(precision, scale) => DataType::Decimal(*precision, *scale),

        // The offsets of these are widened on the way over, which is the one thing the arrays of
        // `polars-array` do not share, and the bytes are then read the same way for all four.
        A::Utf8 | A::LargeUtf8 | A::Binary | A::LargeBinary => DataType::BinaryOffset,
        A::Utf8View => DataType::String,
        A::BinaryView => DataType::Binary,

        // A date is a count of days and a `Date64` a count of milliseconds, which is what a
        // datetime of that unit counts too.
        A::Date32 => DataType::Date,
        A::Date64 => DataType::Datetime(polars_dtype::TimeUnit::Milliseconds, None),
        // A time zone changes no value of a timestamp, so which one it is stays the caller's to
        // read off the type it asked for.
        A::Timestamp(unit, _) => DataType::Datetime(unit.into(), None),
        A::Duration(unit) => DataType::Duration(unit.into()),
        // A time of Polars counts nanoseconds; the other units are the counts they are.
        A::Time64(ArrowTimeUnit::Nanosecond) => DataType::Time,
        A::Time64(_) => DataType::Int64,
        A::Time32(_) => DataType::Int32,

        A::List(field) | A::LargeList(field) => {
            DataType::List(Box::new(datatype_of(field.dtype())?))
        },
        #[cfg(feature = "dtype-array")]
        A::FixedSizeList(field, width) => {
            DataType::Array(Box::new(datatype_of(field.dtype())?), *width)
        },
        #[cfg(feature = "dtype-struct")]
        A::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|field| {
                    Some(polars_dtype::Field::new(
                        field.name.clone(),
                        datatype_of(field.dtype())?,
                    ))
                })
                .collect::<Option<Vec<_>>>()?,
        ),

        // What is left is the Arrow types no array of `polars-array` holds: a dictionary, a map, a
        // union, an interval, and the decimals of another width.
        _ => return None,
    })
}

fn unsupported<T>(from_type: &ArrowDataType, to_type: &ArrowDataType) -> PolarsResult<T> {
    polars_bail!(InvalidOperation: "casting from {from_type:?} to {to_type:?} not supported")
}
