use std::hash::Hash;

use arrow::array::BooleanArray;
use arrow::bitmap::MutableBitmap;
use arrow::legacy::utils::CustomIterTools;
use polars_core::prelude::*;
use polars_core::utils::NoNull;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

pub fn is_last_distinct(s: &Series) -> PolarsResult<BooleanChunked> {
    // fast path.
    if s.len() == 0 {
        return Ok(BooleanChunked::full_null(s.name(), 0));
    } else if s.len() == 1 {
        return Ok(BooleanChunked::new(s.name(), &[true]));
    }

    let s = s.to_physical_repr();

    use DataType::*;
    let out = match s.dtype() {
        Boolean => {
            let ca = s.bool().unwrap();
            is_last_distinct_boolean(ca)
        },
        Binary => {
            let ca = s.binary().unwrap();
            is_last_distinct_bin(ca)
        },
        String => {
            let s = s.cast(&Binary).unwrap();
            return is_last_distinct(&s);
        },
        dt if dt.is_numeric() => {
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                is_last_distinct_numeric(ca)
            })
        },
        #[cfg(feature = "dtype-struct")]
        Struct(_) => return is_last_distinct_struct(&s),
        List(inner) => {
            polars_ensure!(
                !inner.is_nested(),
                InvalidOperation: "`is_last_distinct` on list type is only allowed if the inner type is not nested."
            );
            let ca = s.list().unwrap();
            return is_last_distinct_list(ca);
        },
        dt => polars_bail!(opq = is_last_distinct, dt),
    };
    Ok(out)
}

fn is_last_distinct_boolean(ca: &BooleanChunked) -> BooleanChunked {
    let mut out = MutableBitmap::with_capacity(ca.len());
    out.extend_constant(ca.len(), false);

    if ca.null_count() == ca.len() {
        out.set(ca.len() - 1, true);
    }
    // TODO supports fast path.
    else {
        let mut first_true_found = false;
        let mut first_false_found = false;
        let mut first_null_found = false;
        let mut all_found = false;
        let ca = ca.rechunk();
        let arr = ca.downcast_iter().next().unwrap();
        arr.into_iter()
            .enumerate()
            .rev()
            .find_map(|(idx, val)| match val {
                Some(true) if !first_true_found => {
                    first_true_found = true;
                    all_found &= first_true_found;
                    out.set(idx, true);
                    if all_found {
                        Some(())
                    } else {
                        None
                    }
                },
                Some(false) if !first_false_found => {
                    first_false_found = true;
                    all_found &= first_false_found;
                    out.set(idx, true);
                    if all_found {
                        Some(())
                    } else {
                        None
                    }
                },
                None if !first_null_found => {
                    first_null_found = true;
                    all_found &= first_null_found;
                    out.set(idx, true);
                    if all_found {
                        Some(())
                    } else {
                        None
                    }
                },
                _ => None,
            });
    }

    let arr = BooleanArray::new(ArrowDataType::Boolean, out.into(), None);
    BooleanChunked::with_chunk(ca.name(), arr)
}

fn is_last_distinct_bin(ca: &BinaryChunked) -> BooleanChunked {
    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();
    let mut unique = PlHashSet::new();
    let mut new_ca: BooleanChunked = arr
        .into_iter()
        .rev()
        .map(|opt_v| unique.insert(opt_v))
        .collect_reversed::<NoNull<BooleanChunked>>()
        .into_inner();
    new_ca.rename(ca.name());
    new_ca
}

fn is_last_distinct_numeric<T>(ca: &ChunkedArray<T>) -> BooleanChunked
where
    T: PolarsNumericType,
    T::Native: TotalHash + TotalEq + ToTotalOrd,
    <T::Native as ToTotalOrd>::TotalOrdItem: Hash + Eq,
{
    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();
    let mut unique = PlHashSet::new();
    let mut new_ca: BooleanChunked = arr
        .into_iter()
        .rev()
        .map(|opt_v| unique.insert(opt_v.to_total_ord()))
        .collect_reversed::<NoNull<BooleanChunked>>()
        .into_inner();
    new_ca.rename(ca.name());
    new_ca
}

#[cfg(feature = "dtype-struct")]
fn is_last_distinct_struct(s: &Series) -> PolarsResult<BooleanChunked> {
    let groups = s.group_tuples(true, false)?;
    // SAFETY: all groups have at least a single member
    let last = unsafe { groups.take_group_lasts() };
    let mut out = MutableBitmap::with_capacity(s.len());
    out.extend_constant(s.len(), false);

    for idx in last {
        // Group tuples are always in bounds
        unsafe { out.set_unchecked(idx as usize, true) }
    }

    let arr = BooleanArray::new(ArrowDataType::Boolean, out.into(), None);
    Ok(BooleanChunked::with_chunk(s.name(), arr))
}

fn is_last_distinct_list(ca: &ListChunked) -> PolarsResult<BooleanChunked> {
    let groups = ca.group_tuples(true, false)?;
    // SAFETY: all groups have at least a single member
    let last = unsafe { groups.take_group_lasts() };
    let mut out = MutableBitmap::with_capacity(ca.len());
    out.extend_constant(ca.len(), false);

    for idx in last {
        // Group tuples are always in bounds
        unsafe { out.set_unchecked(idx as usize, true) }
    }

    let arr = BooleanArray::new(ArrowDataType::Boolean, out.into(), None);
    Ok(BooleanChunked::with_chunk(ca.name(), arr))
}
