use polars_core::downcast_as_macro_arg_physical;

use super::*;

fn shift_and_fill_numeric<T>(ca: &ChunkedArray<T>, n: i64, fill_value: AnyValue) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkShiftFill<T, Option<T::Native>>,
{
    let fill_value = fill_value.extract::<T::Native>();
    ca.shift_and_fill(n, fill_value)
}

#[cfg(any(
    feature = "object",
    feature = "dtype-struct",
    feature = "dtype-categorical"
))]
fn shift_and_fill_with_mask(s: &Column, n: i64, fill_value: &Column) -> PolarsResult<Column> {
    use polars_core::export::arrow::array::BooleanArray;
    use polars_core::export::arrow::bitmap::MutableBitmap;

    let mask: BooleanChunked = if n > 0 {
        let len = s.len();
        let mut bits = MutableBitmap::with_capacity(s.len());
        bits.extend_constant(n as usize, false);
        bits.extend_constant(len.saturating_sub(n as usize), true);
        let mask = BooleanArray::from_data_default(bits.into(), None);
        mask.into()
    } else {
        let length = s.len() as i64;
        // n is negative, so subtraction.
        let tipping_point = std::cmp::max(length + n, 0);
        let mut bits = MutableBitmap::with_capacity(s.len());
        bits.extend_constant(tipping_point as usize, true);
        bits.extend_constant(-n as usize, false);
        let mask = BooleanArray::from_data_default(bits.into(), None);
        mask.into()
    };
    s.shift(n).zip_with_same_type(&mask, fill_value)
}

pub(super) fn shift_and_fill(args: &[Column]) -> PolarsResult<Column> {
    let s = &args[0];
    let n_s = &args[1];

    polars_ensure!(
    n_s.len() == 1,
    ComputeError: "n must be a single value."
    );
    let n_s = n_s.cast(&DataType::Int64)?;
    let n = n_s.i64()?;

    if let Some(n) = n.get(0) {
        let logical = s.dtype();
        let physical = s.to_physical_repr();
        let fill_value_s = &args[2];
        let fill_value = fill_value_s.get(0)?;

        use DataType::*;
        match logical {
            Boolean => {
                let ca = s.bool()?;
                let fill_value = match fill_value {
                    AnyValue::Boolean(v) => Some(v),
                    AnyValue::Null => None,
                    v => polars_bail!(ComputeError: "fill value '{}' is not supported", v),
                };
                ca.shift_and_fill(n, fill_value).into_column().cast(logical)
            },
            String => {
                let ca = s.str()?;
                let fill_value = match fill_value {
                    AnyValue::String(v) => Some(v),
                    AnyValue::StringOwned(ref v) => Some(v.as_str()),
                    AnyValue::Null => None,
                    v => polars_bail!(ComputeError: "fill value '{}' is not supported", v),
                };
                ca.shift_and_fill(n, fill_value).into_column().cast(logical)
            },
            List(_) => {
                let ca = s.list()?;
                let fill_value = match fill_value {
                    AnyValue::List(v) => Some(v),
                    AnyValue::Null => None,
                    v => polars_bail!(ComputeError: "fill value '{}' is not supported", v),
                };
                unsafe {
                    ca.shift_and_fill(n, fill_value.as_ref())
                        .into_column()
                        .from_physical_unchecked(logical)
                }
            },
            #[cfg(feature = "object")]
            Object(_, _) => shift_and_fill_with_mask(s, n, fill_value_s),
            #[cfg(feature = "dtype-struct")]
            Struct(_) => shift_and_fill_with_mask(s, n, fill_value_s),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, _) | Enum(_, _) => shift_and_fill_with_mask(s, n, fill_value_s),
            dt if dt.is_primitive_numeric() || dt.is_logical() => {
                macro_rules! dispatch {
                    ($ca:expr, $n:expr, $fill_value:expr) => {{
                        shift_and_fill_numeric($ca, $n, $fill_value).into_column()
                    }};
                }
                let out = downcast_as_macro_arg_physical!(physical, dispatch, n, fill_value);
                unsafe { out.from_physical_unchecked(logical) }
            },
            dt => polars_bail!(opq = shift_and_fill, dt),
        }
    } else {
        Ok(Column::full_null(s.name().clone(), s.len(), s.dtype()))
    }
}

pub fn shift(args: &[Column]) -> PolarsResult<Column> {
    let s = &args[0];
    let n_s = &args[1];
    polars_ensure!(
    n_s.len() == 1,
    ComputeError: "n must be a single value."
    );

    let n_s = n_s.cast(&DataType::Int64)?;
    let n = n_s.i64()?;

    match n.get(0) {
        Some(n) => Ok(s.shift(n)),
        None => Ok(Column::full_null(s.name().clone(), s.len(), s.dtype())),
    }
}
