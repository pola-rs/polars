use arrow::array::{BinaryArray, BinaryViewArray, PrimitiveArray};
use polars_core::downcast_as_macro_arg_physical;
use polars_core::prelude::*;
use polars_utils::total_ord::TotalEq;
use row_encode::encode_rows_unordered;

/// Find the index of the value, or ``None`` if it can't be found.
fn index_of_value<'a, DT, AR>(ca: &'a ChunkedArray<DT>, value: AR::ValueT<'a>) -> Option<usize>
where
    DT: PolarsDataType<Array = AR>,
    AR: StaticArray,
    AR::ValueT<'a>: TotalEq,
{
    let req_value = &value;
    let mut index = 0;
    for chunk in ca.downcast_iter() {
        if chunk.validity().is_some() {
            for maybe_value in chunk.iter() {
                if maybe_value.map(|v| v.tot_eq(req_value)) == Some(true) {
                    return Some(index);
                } else {
                    index += 1;
                }
            }
        } else {
            // A lack of a validity bitmap means there are no nulls, so we
            // can simplify our logic and use a faster code path:
            for value in chunk.values_iter() {
                if value.tot_eq(req_value) {
                    return Some(index);
                } else {
                    index += 1;
                }
            }
        }
    }
    None
}

fn index_of_numeric_value<T>(ca: &ChunkedArray<T>, value: T::Native) -> Option<usize>
where
    T: PolarsNumericType,
{
    index_of_value::<_, PrimitiveArray<T::Native>>(ca, value)
}

fn index_of_bool(ca: &BooleanChunked, value: bool) -> Option<usize> {
    let mut index = 0;
    for chunk in ca.downcast_iter() {
        let num_possible_trues = chunk.values().set_bits();

        // All values are !value.
        if (value && num_possible_trues == 0) || (!value && num_possible_trues == chunk.len()) {
            index += chunk.len();
            continue;
        }

        match chunk.validity() {
            None => {
                // A lack of a validity bitmap means there are no nulls, so we
                // can simplify our logic and use a faster code path:
                let n = if value {
                    chunk.values().leading_zeros()
                } else {
                    chunk.values().leading_ones()
                };

                if n != chunk.len() {
                    return Some(index + n);
                }
                index += n;
            },
            Some(validity) => {
                // All values are none.
                if validity.set_bits() == 0 {
                    index += chunk.len();
                    continue;
                }

                let mask = if value { 0 } else { u64::MAX };
                let mut validity = validity.fast_iter_u56();
                let mut values = chunk.values().fast_iter_u56();
                for (is_null, value) in validity.by_ref().zip(values.by_ref()) {
                    let eq_mask = is_null & (value ^ mask);
                    if eq_mask != 0 {
                        return Some(index + eq_mask.trailing_zeros() as usize);
                    }
                    index += 56;
                }

                let (is_null, l1) = validity.remainder();
                let (value, l2) = values.remainder();
                assert_eq!(l1, l2);

                let eq_mask = is_null & (value ^ mask);
                if eq_mask != 0 {
                    return Some(index + eq_mask.trailing_zeros() as usize);
                }
                index += l1;
            },
        }
    }
    None
}

/// Try casting the value to the correct type, then call
/// index_of_numeric_value().
macro_rules! try_index_of_numeric_ca {
    ($ca:expr, $value:expr) => {{
        let ca = $ca;
        let value = $value;
        // extract() returns None if casting failed, so consider an extract()
        // failure as not finding the value. Nulls should have been handled
        // earlier.
        let value = value.into_value().to_physical().extract().unwrap();
        index_of_numeric_value(ca, value)
    }};
}

/// Find the index of a given value (the first and only entry in `value_series`)
/// within the series.
pub fn index_of(series: &Series, needle: Scalar) -> PolarsResult<Option<usize>> {
    polars_ensure!(
        series.dtype() == needle.dtype(),
        InvalidOperation: "Cannot perform index_of with mismatching datatypes: {:?} and {:?}",
        series.dtype(),
        needle.dtype(),
    );

    if series.is_empty() {
        return Ok(None);
    }

    // Series is not null, and the value is null:
    if needle.is_null() {
        let null_count = series.null_count();
        if null_count == 0 {
            return Ok(None);
        } else if null_count == series.len() {
            return Ok(Some(0));
        }

        let mut index = 0;
        for chunk in series.chunks() {
            let length = chunk.len();
            if let Some(bitmap) = chunk.validity() {
                let leading_ones = bitmap.leading_ones();
                if leading_ones < length {
                    return Ok(Some(index + leading_ones));
                }
            } else {
                index += length;
            }
        }
        return Ok(None);
    }

    use DataType as DT;
    match series.dtype().to_physical() {
        DT::Null => unreachable!("handled above"),
        DT::Boolean => Ok(index_of_bool(
            series.bool()?,
            needle.value().extract_bool().unwrap(),
        )),
        dt if dt.is_primitive_numeric() => {
            let series = series.to_physical_repr();
            Ok(downcast_as_macro_arg_physical!(
                series,
                try_index_of_numeric_ca,
                needle
            ))
        },
        DT::String => Ok(index_of_value::<_, BinaryViewArray>(
            &series.str()?.as_binary(),
            needle.value().extract_str().unwrap().as_bytes(),
        )),
        DT::Binary => Ok(index_of_value::<_, BinaryViewArray>(
            series.binary()?,
            needle.value().extract_bytes().unwrap(),
        )),
        DT::BinaryOffset => Ok(index_of_value::<_, BinaryArray<i64>>(
            series.binary_offset()?,
            needle.value().extract_bytes().unwrap(),
        )),
        DT::Array(_, _) | DT::List(_) | DT::Struct(_) => {
            // For non-numeric dtypes, we convert to row-encoding, which essentially has
            // us searching the physical representation of the data as a series of
            // bytes.
            let value_as_column = Column::new_scalar(PlSmallStr::EMPTY, needle, 1);
            let value_as_row_encoded_ca = encode_rows_unordered(&[value_as_column])?;
            let value = value_as_row_encoded_ca
                .first()
                .expect("Shouldn't have nulls in a row-encoded result");
            let ca = encode_rows_unordered(&[series.clone().into_column()])?;
            Ok(index_of_value::<_, BinaryArray<i64>>(&ca, value))
        },

        DT::UInt8
        | DT::UInt16
        | DT::UInt32
        | DT::UInt64
        | DT::Int8
        | DT::Int16
        | DT::Int32
        | DT::Int64
        | DT::Int128
        | DT::Float32
        | DT::Float64 => unreachable!("primitive numeric"),

        // to_physical
        #[cfg(feature = "dtype-decimal")]
        DT::Decimal(..) => unreachable!(),
        #[cfg(feature = "dtype-categorical")]
        DT::Categorical(..) | DT::Enum(..) => unreachable!(),
        DT::Date | DT::Datetime(..) | DT::Duration(..) | DT::Time => unreachable!(),

        DT::Object(_) | DT::Unknown(_) => polars_bail!(op = "index_of", series.dtype()),
    }
}
