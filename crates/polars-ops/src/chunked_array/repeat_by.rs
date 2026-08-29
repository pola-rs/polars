use std::borrow::Cow;

use arrow::array::builder::{ArrayBuilder, ShareStrategy, make_builder};
use arrow::array::{Array, IntoBoxedArray, ListArray, NullArray};
use arrow::bitmap::BitmapBuilder;
use arrow::offset::Offsets;
use arrow::pushable::Pushable;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_error::PolarsContext;
use polars_error::constants::LENGTH_LIMIT_MSG;
use polars_utils::broadcast::broadcast_len;

type LargeListArray = ListArray<i64>;

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

fn repeat_by_primitive<T>(ca: &ChunkedArray<T>, by: &IdxCa) -> PolarsResult<ListChunked>
where
    T: PolarsNumericType,
{
    let (ca, by) = broadcast_args(ca, by)?;

    Ok(arity::binary(&ca, &by, |arr, by| {
        let iter = arr.into_iter().zip(by).map(|(opt_v, opt_by)| {
            opt_by.map(|by| std::iter::repeat_n(opt_v.copied(), *by as usize))
        });

        // SAFETY: length of iter is trusted.
        unsafe {
            LargeListArray::from_iter_primitive_trusted_len(
                iter,
                T::get_static_dtype().to_arrow(CompatLevel::newest()),
            )
        }
    }))
}

fn repeat_by_bool(ca: &BooleanChunked, by: &IdxCa) -> PolarsResult<ListChunked> {
    let (ca, by) = broadcast_args(ca, by)?;

    Ok(arity::binary(&ca, &by, |arr, by| {
        let iter = arr
            .into_iter()
            .zip(by)
            .map(|(opt_v, opt_by)| opt_by.map(|by| std::iter::repeat_n(opt_v, *by as usize)));

        // SAFETY: length of iter is trusted.
        unsafe { LargeListArray::from_iter_bool_trusted_len(iter) }
    }))
}

fn repeat_by_binary(ca: &BinaryChunked, by: &IdxCa) -> PolarsResult<ListChunked> {
    let (ca, by) = broadcast_args(ca, by)?;

    Ok(arity::binary(&ca, &by, |arr, by| {
        let iter = arr
            .into_iter()
            .zip(by)
            .map(|(opt_v, opt_by)| opt_by.map(|by| std::iter::repeat_n(opt_v, *by as usize)));

        // SAFETY: length of iter is trusted.
        unsafe { LargeListArray::from_iter_binary_trusted_len(iter, ca.len()) }
    }))
}

fn repeat_by_null(ca: &NullChunked, by: &IdxCa) -> PolarsResult<ListChunked> {
    // All values are null, so only the repeat counts have to be broadcast.
    let len = broadcast_len([ca.len(), by.len()]).context("repeat_by")?;
    let by = by.broadcast_to(len)?;

    let arr_length = by.iter().flatten().map(|x| x as usize).sum();
    let arr = NullArray::new(ArrowDataType::Null, arr_length);

    let mut validity = BitmapBuilder::with_capacity(by.len());
    let mut offsets = Offsets::<i64>::with_capacity(by.len());
    for n_repeat in by.iter() {
        validity.push(n_repeat.is_some());
        if let Some(repeats) = n_repeat {
            offsets.push(repeats as usize);
        } else {
            offsets.push_null();
        }
    }

    let array = LargeListArray::new(
        ListArray::<i64>::default_datatype(arr.dtype().clone()),
        offsets.into(),
        arr.into_boxed(),
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

fn repeat_by_generic<T: PolarsDataType>(
    ca: &ChunkedArray<T>,
    by: &IdxCa,
) -> PolarsResult<ListChunked>
where
    ChunkedArray<T>: ChunkExpandAtIndex<T>,
{
    let (ca, by) = broadcast_args(ca, by)?;
    let mut builder = make_builder(&ca.dtype().to_arrow(CompatLevel::newest()));
    Ok(arity::binary(&ca, &by, |arr, by| {
        let arr_length = by.iter().flatten().map(|x| *x as usize).sum();
        builder.reserve(arr_length);

        let mut validity = BitmapBuilder::with_capacity(by.len());
        let mut offsets = Offsets::<i64>::with_capacity(by.len());
        for (idx, n_repeat) in by.iter().enumerate() {
            validity.push(n_repeat.is_some());
            if let Some(repeats) = n_repeat {
                offsets.push(*repeats as usize);
                builder.subslice_extend_repeated(
                    arr,
                    idx,
                    1,
                    *repeats as usize,
                    ShareStrategy::Always,
                );
            } else {
                offsets.push_null();
            }
        }

        let repeated_values = builder.freeze_reset();
        LargeListArray::new(
            ListArray::<i64>::default_datatype(arr.dtype().clone()),
            offsets.into(),
            repeated_values,
            validity.into_opt_validity(),
        )
    }))
}

pub fn repeat_by(s: &Series, by: &IdxCa) -> PolarsResult<ListChunked> {
    let s_phys = s.to_physical_repr();
    use DataType as D;
    let out = match s_phys.dtype() {
        D::Null => repeat_by_null(s_phys.null().unwrap(), by),
        D::Boolean => repeat_by_bool(s_phys.bool().unwrap(), by),
        D::String => {
            let ca = s_phys.str().unwrap();
            repeat_by_binary(&ca.as_binary(), by)
                .and_then(|ca| ca.apply_to_inner(&|s| unsafe { s.cast_unchecked(&D::String) }))
        },
        D::Binary => repeat_by_binary(s_phys.binary().unwrap(), by),
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s_phys.as_ref().as_ref().as_ref();
                repeat_by_primitive(ca, by)
            })
        },
        D::List(_) => repeat_by_generic(s_phys.list().unwrap(), by),
        #[cfg(feature = "dtype-struct")]
        D::Struct(_) => repeat_by_generic(s_phys.struct_().unwrap(), by),
        #[cfg(feature = "dtype-array")]
        D::Array(_, _) => repeat_by_generic(s_phys.array().unwrap(), by),
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
