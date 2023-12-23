use arrow::array::BooleanArray;
use arrow::bitmap::MutableBitmap;
use polars_core::prelude::*;
use polars_core::with_match_physical_integer_polars_type;
use polars_utils::total_ord::{TotalEq, TotalHash, TotalOrdWrap};

// If invert is true then this is an `is_duplicated`.
fn is_unique_ca<'a, T>(ca: &'a ChunkedArray<T>, invert: bool) -> BooleanChunked
where
    T: PolarsDataType,
    &'a ChunkedArray<T>: IntoIterator,
    <<&'a ChunkedArray<T> as IntoIterator>::IntoIter as IntoIterator>::Item: TotalHash + TotalEq,
{
    let len = ca.len();
    let mut idx_key = PlHashMap::new();

    // Instead of group_tuples, which allocates a full Vec per group, we now
    // just toggle a boolean that's false if a group has multiple entries.
    ca.into_iter().enumerate().for_each(|(idx, key)| {
        idx_key
            .entry(TotalOrdWrap(key))
            .and_modify(|v: &mut (IdxSize, bool)| v.1 = false)
            .or_insert((idx as IdxSize, true));
    });

    let unique_idx = idx_key
        .into_iter()
        .filter_map(|(_k, v)| if v.1 { Some(v.0) } else { None });

    let (default, setter) = if invert { (true, false) } else { (false, true) };
    let mut values = MutableBitmap::with_capacity(len);
    values.extend_constant(len, default);
    for idx in unique_idx {
        unsafe { values.set_unchecked(idx as usize, setter) }
    }
    let arr = BooleanArray::from_data_default(values.into(), None);
    BooleanChunked::with_chunk(ca.name(), arr)
}

fn dispatcher(s: &Series, invert: bool) -> PolarsResult<BooleanChunked> {
    let s = s.to_physical_repr();
    use DataType::*;
    let out = match s.dtype() {
        Boolean => {
            let ca = s.bool().unwrap();
            is_unique_ca(ca, invert)
        },
        Binary => {
            let ca = s.binary().unwrap();
            is_unique_ca(ca, invert)
        },
        String => {
            let s = s.cast(&Binary).unwrap();
            let ca = s.binary().unwrap();
            is_unique_ca(ca, invert)
        },
        Float32 => {
            let ca = s.f32().unwrap();
            is_unique_ca(ca, invert)
        },
        Float64 => {
            let ca = s.f64().unwrap();
            is_unique_ca(ca, invert)
        },
        #[cfg(feature = "dtype-struct")]
        Struct(_) => {
            let ca = s.struct_().unwrap().clone();
            let df = ca.unnest();
            return if invert {
                df.is_duplicated()
            } else {
                df.is_unique()
            };
        },
        dt if dt.is_numeric() => {
            with_match_physical_integer_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                is_unique_ca(ca, invert)
            })
        },
        dt => polars_bail!(opq = is_unique, dt),
    };
    Ok(out)
}

pub fn is_unique(s: &Series) -> PolarsResult<BooleanChunked> {
    dispatcher(s, false)
}

pub fn is_duplicated(s: &Series) -> PolarsResult<BooleanChunked> {
    dispatcher(s, true)
}
