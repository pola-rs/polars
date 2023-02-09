use std::hash::Hash;

use arrow::array::BooleanArray;
use arrow::bitmap::MutableBitmap;
use polars_arrow::utils::CustomIterTools;
use polars_core::prelude::*;
use polars_core::with_match_physical_integer_polars_type;

use crate::series::ops::arg_min_max::arg_max;

fn is_first_numeric<T>(ca: &ChunkedArray<T>) -> BooleanChunked
where
    T: PolarsNumericType,
    T::Native: Hash + Eq,
{
    let mut unique = PlHashSet::new();
    let chunks = ca
        .downcast_iter()
        .map(|arr| {
            let mask: BooleanArray = arr
                .into_iter()
                .map(|opt_v| unique.insert(opt_v))
                .collect_trusted();
            Box::new(mask) as ArrayRef
        })
        .collect();

    unsafe { BooleanChunked::from_chunks(ca.name(), chunks) }
}

#[cfg(feature = "dtype-binary")]
fn is_first_bin(ca: &BinaryChunked) -> BooleanChunked {
    let mut unique = PlHashSet::new();
    let chunks = ca
        .downcast_iter()
        .map(|arr| {
            let mask: BooleanArray = arr
                .into_iter()
                .map(|opt_v| unique.insert(opt_v))
                .collect_trusted();
            Box::new(mask) as ArrayRef
        })
        .collect();

    unsafe { BooleanChunked::from_chunks(ca.name(), chunks) }
}

fn is_first_boolean(ca: &BooleanChunked) -> BooleanChunked {
    let mut out = MutableBitmap::with_capacity(ca.len());
    out.extend_constant(ca.len(), false);
    if let Some(index) = arg_max(ca) {
        out.set(index, true)
    }
    if let Some(index) = ca.first_non_null() {
        out.set(index, true)
    }

    let chunks =
        vec![Box::new(BooleanArray::new(ArrowDataType::Boolean, out.into(), None)) as ArrayRef];
    unsafe { BooleanChunked::from_chunks(ca.name(), chunks) }
}

#[cfg(feature = "dtype-struct")]
fn is_first_struct(s: &Series) -> PolarsResult<BooleanChunked> {
    let groups = s.group_tuples(true, false)?;
    let first = groups.take_group_firsts();
    let mut out = MutableBitmap::with_capacity(s.len());
    out.extend_constant(s.len(), false);

    for idx in first {
        // Group tuples are always in bounds
        unsafe { out.set_unchecked(idx as usize, true) }
    }
    let chunks =
        vec![Box::new(BooleanArray::new(ArrowDataType::Boolean, out.into(), None)) as ArrayRef];
    Ok(unsafe { BooleanChunked::from_chunks(s.name(), chunks) })
}

pub fn is_first(s: &Series) -> PolarsResult<BooleanChunked> {
    let s = s.to_physical_repr();

    use DataType::*;
    let out = match s.dtype() {
        Boolean => {
            let ca = s.bool().unwrap();
            is_first_boolean(ca)
        }
        #[cfg(feature = "dtype-binary")]
        Binary => {
            let ca = s.binary().unwrap();
            is_first_bin(ca)
        }
        #[cfg(feature = "dtype-binary")]
        Utf8 => {
            let s = s.cast(&Binary).unwrap();
            return is_first(&s);
        }
        Float32 => {
            let ca = s.bit_repr_small();
            is_first_numeric(&ca)
        }
        Float64 => {
            let ca = s.bit_repr_large();
            is_first_numeric(&ca)
        }
        dt if dt.is_numeric() => {
            with_match_physical_integer_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                is_first_numeric(ca)
            })
        }
        #[cfg(feature = "dtype-struct")]
        Struct(_) => return is_first_struct(&s),
        dt => panic!("dtype {dt} not supported in 'is_first' operation"),
    };
    Ok(out)
}
