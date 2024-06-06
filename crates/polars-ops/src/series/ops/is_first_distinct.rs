use std::hash::Hash;

use arrow::array::BooleanArray;
use arrow::bitmap::MutableBitmap;
use arrow::legacy::bit_util::*;
use arrow::legacy::utils::CustomIterTools;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};
fn is_first_distinct_numeric<T>(ca: &ChunkedArray<T>) -> BooleanChunked
where
    T: PolarsNumericType,
    T::Native: TotalHash + TotalEq + ToTotalOrd,
    <T::Native as ToTotalOrd>::TotalOrdItem: Hash + Eq,
{
    let mut unique = PlHashSet::new();
    let chunks = ca.downcast_iter().map(|arr| -> BooleanArray {
        arr.into_iter()
            .map(|opt_v| unique.insert(opt_v.to_total_ord()))
            .collect_trusted()
    });

    BooleanChunked::from_chunk_iter(ca.name(), chunks)
}

fn is_first_distinct_bin(ca: &BinaryChunked) -> BooleanChunked {
    let mut unique = PlHashSet::new();
    let chunks = ca.downcast_iter().map(|arr| -> BooleanArray {
        arr.into_iter()
            .map(|opt_v| unique.insert(opt_v))
            .collect_trusted()
    });

    BooleanChunked::from_chunk_iter(ca.name(), chunks)
}

fn is_first_distinct_boolean(ca: &BooleanChunked) -> BooleanChunked {
    let mut out = MutableBitmap::with_capacity(ca.len());
    out.extend_constant(ca.len(), false);

    if ca.null_count() == ca.len() {
        out.set(0, true);
    } else {
        let ca = ca.rechunk();
        let arr = ca.downcast_iter().next().unwrap();
        if ca.null_count() == 0 {
            let (true_index, false_index) =
                find_first_true_false_no_null(arr.values().chunks::<u64>());
            if let Some(idx) = true_index {
                out.set(idx, true)
            }
            if let Some(idx) = false_index {
                out.set(idx, true)
            }
        } else {
            let (true_index, false_index, null_index) = find_first_true_false_null(
                arr.values().chunks::<u64>(),
                arr.validity().unwrap().chunks::<u64>(),
            );
            if let Some(idx) = true_index {
                out.set(idx, true)
            }
            if let Some(idx) = false_index {
                out.set(idx, true)
            }
            if let Some(idx) = null_index {
                out.set(idx, true)
            }
        }
    }
    let arr = BooleanArray::new(ArrowDataType::Boolean, out.into(), None);
    BooleanChunked::with_chunk(ca.name(), arr)
}

#[cfg(feature = "dtype-struct")]
fn is_first_distinct_struct(s: &Series) -> PolarsResult<BooleanChunked> {
    let groups = s.group_tuples(true, false)?;
    let first = groups.take_group_firsts();
    let mut out = MutableBitmap::with_capacity(s.len());
    out.extend_constant(s.len(), false);

    for idx in first {
        // Group tuples are always in bounds
        unsafe { out.set_unchecked(idx as usize, true) }
    }

    let arr = BooleanArray::new(ArrowDataType::Boolean, out.into(), None);
    Ok(BooleanChunked::with_chunk(s.name(), arr))
}

fn is_first_distinct_list(ca: &ListChunked) -> PolarsResult<BooleanChunked> {
    let groups = ca.group_tuples(true, false)?;
    let first = groups.take_group_firsts();
    let mut out = MutableBitmap::with_capacity(ca.len());
    out.extend_constant(ca.len(), false);

    for idx in first {
        // Group tuples are always in bounds
        unsafe { out.set_unchecked(idx as usize, true) }
    }

    let arr = BooleanArray::new(ArrowDataType::Boolean, out.into(), None);
    Ok(BooleanChunked::with_chunk(ca.name(), arr))
}

pub fn is_first_distinct(s: &Series) -> PolarsResult<BooleanChunked> {
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
            is_first_distinct_boolean(ca)
        },
        Binary => {
            let ca = s.binary().unwrap();
            is_first_distinct_bin(ca)
        },
        String => {
            let s = s.cast(&Binary).unwrap();
            return is_first_distinct(&s);
        },
        dt if dt.is_numeric() => {
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                is_first_distinct_numeric(ca)
            })
        },
        #[cfg(feature = "dtype-struct")]
        Struct(_) => return is_first_distinct_struct(&s),
        List(inner) => {
            polars_ensure!(
                !inner.is_nested(),
                InvalidOperation: "`is_first_distinct` on list type is only allowed if the inner type is not nested."
            );
            let ca = s.list().unwrap();
            return is_first_distinct_list(ca);
        },
        dt => polars_bail!(opq = is_first_distinct, dt),
    };
    Ok(out)
}
