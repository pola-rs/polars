use polars_core::downcast_as_macro_arg_physical;

use super::*;

fn shift_and_fill_numeric<T>(
    ca: &ChunkedArray<T>,
    periods: i64,
    fill_value: AnyValue,
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkShiftFill<T, Option<T::Native>>,
{
    let fill_value = fill_value.extract::<T::Native>();
    ca.shift_and_fill(periods, fill_value)
}

#[cfg(any(
    feature = "object",
    feature = "dtype-struct",
    feature = "dtype-categorical"
))]
fn shift_and_fill_with_mask(s: &Series, periods: i64, fill_value: &Series) -> PolarsResult<Series> {
    use polars_core::export::arrow::array::BooleanArray;
    use polars_core::export::arrow::bitmap::MutableBitmap;

    let mask: BooleanChunked = if periods > 0 {
        let len = s.len();
        let mut bits = MutableBitmap::with_capacity(s.len());
        bits.extend_constant(periods as usize, false);
        bits.extend_constant(len.saturating_sub(periods as usize), true);
        let mask = BooleanArray::from_data_default(bits.into(), None);
        mask.into()
    } else {
        let length = s.len() as i64;
        // periods is negative, so subtraction.
        let tipping_point = std::cmp::max(length + periods, 0);
        let mut bits = MutableBitmap::with_capacity(s.len());
        bits.extend_constant(tipping_point as usize, true);
        bits.extend_constant(-periods as usize, false);
        let mask = BooleanArray::from_data_default(bits.into(), None);
        mask.into()
    };
    s.shift(periods).zip_with_same_type(&mask, fill_value)
}

pub(super) fn shift_and_fill(args: &mut [Series], periods: i64) -> PolarsResult<Series> {
    let s = &args[0];
    let logical = s.dtype();
    let physical = s.to_physical_repr();
    let fill_value_s = &args[1];
    let fill_value = fill_value_s.get(0).unwrap();

    use DataType::*;
    match logical {
        Boolean => {
            let ca = s.bool().unwrap();
            let fill_value = match fill_value {
                AnyValue::Boolean(v) => Some(v),
                AnyValue::Null => None,
                _ => unimplemented!(),
            };
            ca.shift_and_fill(periods, fill_value)
                .into_series()
                .cast(logical)
        }
        Utf8 => {
            let ca = s.utf8().unwrap();
            let fill_value = match fill_value {
                AnyValue::Utf8(v) => Some(v),
                AnyValue::Null => None,
                _ => unimplemented!(),
            };
            ca.shift_and_fill(periods, fill_value)
                .into_series()
                .cast(logical)
        }
        List(_) => {
            let ca = s.list().unwrap();
            let fill_value = match fill_value {
                AnyValue::List(v) => Some(v),
                AnyValue::Null => None,
                _ => unimplemented!(),
            };
            ca.shift_and_fill(periods, fill_value.as_ref())
                .into_series()
                .cast(logical)
        }
        #[cfg(feature = "object")]
        Object(_) => shift_and_fill_with_mask(s, periods, fill_value_s),
        #[cfg(feature = "dtype-struct")]
        Struct(_) => shift_and_fill_with_mask(s, periods, fill_value_s),
        #[cfg(feature = "dtype-categorical")]
        Categorical(_) => shift_and_fill_with_mask(s, periods, fill_value_s),
        dt if dt.is_numeric() || dt.is_logical() => {
            macro_rules! dispatch {
                ($ca:expr, $periods:expr, $fill_value:expr) => {{
                    shift_and_fill_numeric($ca, $periods, $fill_value).into_series()
                }};
            }

            let out = downcast_as_macro_arg_physical!(physical, dispatch, periods, fill_value);
            out.cast(logical)
        }
        _ => {
            unimplemented!()
        }
    }
}
