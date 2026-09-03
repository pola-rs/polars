use std::borrow::Cow;

use arrow::bitmap::BitmapBuilder;
use polars_array::builder::{PlArrayBuilder, ShareStrategy, builder_like};
use polars_buffer::Buffer;
use polars_core::prelude::*;
use polars_core::utils::align_chunks_binary;
use polars_core::with_match_physical_numeric_polars_type;
use polars_error::PolarsContext;
use polars_error::constants::LENGTH_LIMIT_MSG;
use polars_utils::broadcast::broadcast_len;
use polars_utils::index::idxsize_to_u64;

/// Broadcasts the values and the repeat counts to their common length.
fn broadcast_args<'a, 'b, T: PolarsDataType>(
    ca: &'a ChunkedArray<T>,
    by: &'b IdxCa,
) -> PolarsResult<(Cow<'a, ChunkedArray<T>>, Cow<'b, IdxCa>)>
where
    ChunkedArray<T>: ChunkExpandAtIndex<T>,
{
    let len = broadcast_len([ca.len(), by.len()]).context("repeat_by")?;
    Ok((ca.broadcast_to(len)?, by.broadcast_to(len)?))
}

/// One chunk of the result: element `i` of `arr` repeated `by[i]` times, as one list. A null
/// repeat count makes the whole list null.
fn repeat_chunk(arr: &dyn PlArray, by: &PlPrimitiveArray<IdxSize>) -> PlListArray {
    // TODO(polars-array-scalar): the repeated values are written out one element at a time, so a
    // scalar chunk is materialized here rather than the lists built as one scalar array.
    let mut values = builder_like(arr);
    let mut offsets = Vec::with_capacity(by.len() + 1);
    offsets.push(0);
    let mut validity = BitmapBuilder::with_capacity(by.len());

    for (idx, n_repeat) in by.iter().enumerate() {
        validity.push(n_repeat.is_some());
        if let Some(repeats) = n_repeat {
            values.subslice_extend_repeated(arr, idx, 1, repeats as usize, ShareStrategy::Always);
        }
        offsets.push(values.len() as u64);
    }

    PlListArray::new(
        PlArrayBuilder::freeze(values),
        Buffer::from(offsets),
        by.len(),
        validity.into_opt_validity(),
    )
}

fn repeat_by_impl<T: PolarsDataType>(ca: &ChunkedArray<T>, by: &IdxCa) -> PolarsResult<ListChunked>
where
    ChunkedArray<T>: ChunkExpandAtIndex<T>,
{
    let (ca, by) = broadcast_args(ca, by)?;
    let (ca, by) = align_chunks_binary(ca.as_ref(), by.as_ref());

    let chunks = ca
        .downcast_iter()
        .zip(by.downcast_iter())
        .map(|(arr, by)| repeat_chunk(arr, by).into_boxed())
        .collect();

    // The chunks carry no logical type of their own, so the inner one is named here.
    let dtype = DataType::List(Box::new(ca.dtype().clone()));
    Ok(unsafe { ListChunked::from_chunks_and_dtype(ca.name().clone(), chunks, dtype) })
}

/// Every element being null, only the repeat counts decide the shape of the lists.
fn repeat_by_null(ca: &NullChunked, by: &IdxCa) -> PolarsResult<ListChunked> {
    let len = broadcast_len([ca.len(), by.len()]).context("repeat_by")?;
    let by = by.broadcast_to(len)?;

    let mut offsets = Vec::with_capacity(by.len() + 1);
    offsets.push(0);
    let mut validity = BitmapBuilder::with_capacity(by.len());
    let mut offset = 0;

    for n_repeat in by.iter() {
        validity.push(n_repeat.is_some());
        offset += idxsize_to_u64(n_repeat.unwrap_or(0));
        offsets.push(offset);
    }

    // A null array is its length and nothing else, so the values cost `O(1)` however many nulls
    // the lists reach.
    let array = PlListArray::new(
        PlNullArray::new(offset as usize).into_boxed(),
        Buffer::from(offsets),
        by.len(),
        validity.into_opt_validity(),
    );

    Ok(unsafe {
        ListChunked::from_chunks_and_dtype(
            ca.name().clone(),
            vec![array.into_boxed()],
            DataType::List(Box::new(DataType::Null)),
        )
    })
}

pub fn repeat_by(s: &Series, by: &IdxCa) -> PolarsResult<ListChunked> {
    let s_phys = s.to_physical_repr();
    use DataType as D;
    let out = match s_phys.dtype() {
        D::Null => repeat_by_null(s_phys.null().unwrap(), by),
        D::Boolean => repeat_by_impl(s_phys.bool().unwrap(), by),
        D::String => {
            let ca = s_phys.str().unwrap();
            repeat_by_impl(&ca.as_binary(), by)
                .and_then(|ca| ca.apply_to_inner(&|s| unsafe { s.cast_unchecked(&D::String) }))
        },
        D::Binary => repeat_by_impl(s_phys.binary().unwrap(), by),
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s_phys.as_ref().as_ref().as_ref();
                repeat_by_impl(ca, by)
            })
        },
        D::List(_) => repeat_by_impl(s_phys.list().unwrap(), by),
        #[cfg(feature = "dtype-struct")]
        D::Struct(_) => repeat_by_impl(s_phys.struct_().unwrap(), by),
        #[cfg(feature = "dtype-array")]
        D::Array(_, _) => repeat_by_impl(s_phys.array().unwrap(), by),
        _ => polars_bail!(opq = repeat_by, s.dtype()),
    };
    out.and_then(|ca| {
        // `ca.len()` is the number of output rows (one list per input row), not the total
        // number of repeated elements, which is what can actually overflow `IdxSize`.
        polars_ensure!(ca.inner_length() < IdxSize::MAX as usize, ComputeError: "{LENGTH_LIMIT_MSG}");
        let logical_type = s.dtype();
        ca.apply_to_inner(&|s| unsafe { s.from_physical_unchecked(logical_type) })
    })
}
