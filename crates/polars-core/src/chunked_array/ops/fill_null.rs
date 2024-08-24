use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::legacy::kernels::set::set_at_nulls;
use bytemuck::Zeroable;
use num_traits::{Bounded, NumCast, One, Zero};
use polars_utils::itertools::Itertools;

use crate::prelude::*;

fn err_fill_null() -> PolarsError {
    polars_err!(ComputeError: "could not determine the fill value")
}

impl Series {
    /// Replace None values with one of the following strategies:
    /// * Forward fill (replace None with the previous value)
    /// * Backward fill (replace None with the next value)
    /// * Mean fill (replace None with the mean of the whole array)
    /// * Min fill (replace None with the minimum of the whole array)
    /// * Max fill (replace None with the maximum of the whole array)
    /// * Zero fill (replace None with the value zero)
    /// * One fill (replace None with the value one)
    /// * MinBound fill (replace with the minimum of that data type)
    /// * MaxBound fill (replace with the maximum of that data type)
    ///
    /// *NOTE: If you want to fill the Nones with a value use the
    /// [`fill_null` operation on `ChunkedArray<T>`](crate::chunked_array::ops::ChunkFillNullValue)*.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example() -> PolarsResult<()> {
    ///     let s = Series::new("some_missing", &[Some(1), None, Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Forward(None))?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Backward(None))?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(2), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Min)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Max)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(2), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Mean)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Zero)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(0), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::One)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::MinBound)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(-2147483648), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::MaxBound)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(2147483647), Some(2)]);
    ///
    ///     Ok(())
    /// }
    /// example();
    /// ```
    pub fn fill_null(&self, strategy: FillNullStrategy) -> PolarsResult<Series> {
        // Nothing to fill.
        let nc = self.null_count();
        if nc == 0
            || (nc == self.len()
                && matches!(
                    strategy,
                    FillNullStrategy::Forward(_)
                        | FillNullStrategy::Backward(_)
                        | FillNullStrategy::Max
                        | FillNullStrategy::Min
                        | FillNullStrategy::MaxBound
                        | FillNullStrategy::MinBound
                        | FillNullStrategy::Mean
                ))
        {
            return Ok(self.clone());
        }

        let physical_type = self.dtype().to_physical();

        match strategy {
            FillNullStrategy::Forward(None) if !physical_type.is_numeric() => {
                fill_forward_gather(self)
            },
            FillNullStrategy::Forward(Some(limit)) => fill_forward_gather_limit(self, limit),
            FillNullStrategy::Backward(None) if !physical_type.is_numeric() => {
                fill_backward_gather(self)
            },
            FillNullStrategy::Backward(Some(limit)) => fill_backward_gather_limit(self, limit),
            _ => {
                let logical_type = self.dtype();
                let s = self.to_physical_repr();
                use DataType::*;
                let out = match s.dtype() {
                    Boolean => fill_null_bool(s.bool().unwrap(), strategy),
                    String => {
                        let s = unsafe { s.cast_unchecked(&Binary)? };
                        let out = s.fill_null(strategy)?;
                        return unsafe { out.cast_unchecked(&String) };
                    },
                    Binary => {
                        let ca = s.binary().unwrap();
                        fill_null_binary(ca, strategy).map(|ca| ca.into_series())
                    },
                    dt if dt.is_numeric() => {
                        with_match_physical_numeric_polars_type!(dt, |$T| {
                            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                                fill_null_numeric(ca, strategy).map(|ca| ca.into_series())
                        })
                    },
                    dt => {
                        polars_bail!(InvalidOperation: "fill null strategy not yet supported for dtype: {}", dt)
                    },
                }?;
                unsafe { out.cast_unchecked(logical_type) }
            },
        }
    }
}

fn fill_forward_numeric<'a, T, I>(ca: &'a ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsDataType,
    &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
    I: TrustedLen + Iterator<Item = Option<T::Physical<'a>>>,
    T::ZeroablePhysical<'a>: Copy,
{
    // Compute values.
    let values: Vec<T::ZeroablePhysical<'a>> = ca
        .into_iter()
        .scan(T::ZeroablePhysical::zeroed(), |prev, v| {
            *prev = v.map(|v| v.into()).unwrap_or(*prev);
            Some(*prev)
        })
        .collect_trusted();

    // Compute bitmask.
    let num_start_nulls = ca.first_non_null().unwrap_or(ca.len());
    let mut bm = MutableBitmap::with_capacity(ca.len());
    bm.extend_constant(num_start_nulls, false);
    bm.extend_constant(ca.len() - num_start_nulls, true);
    ChunkedArray::from_chunk_iter_like(
        ca,
        [
            T::Array::from_zeroable_vec(values, ca.dtype().to_arrow(CompatLevel::newest()))
                .with_validity_typed(Some(bm.into())),
        ],
    )
}

fn fill_backward_numeric<'a, T, I>(ca: &'a ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsDataType,
    &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
    I: TrustedLen + Iterator<Item = Option<T::Physical<'a>>> + DoubleEndedIterator,
    T::ZeroablePhysical<'a>: Copy,
{
    // Compute values.
    let values: Vec<T::ZeroablePhysical<'a>> = ca
        .into_iter()
        .rev()
        .scan(T::ZeroablePhysical::zeroed(), |prev, v| {
            *prev = v.map(|v| v.into()).unwrap_or(*prev);
            Some(*prev)
        })
        .collect_reversed();

    // Compute bitmask.
    let num_end_nulls = ca
        .last_non_null()
        .map(|i| ca.len() - 1 - i)
        .unwrap_or(ca.len());
    let mut bm = MutableBitmap::with_capacity(ca.len());
    bm.extend_constant(ca.len() - num_end_nulls, true);
    bm.extend_constant(num_end_nulls, false);
    ChunkedArray::from_chunk_iter_like(
        ca,
        [
            T::Array::from_zeroable_vec(values, ca.dtype().to_arrow(CompatLevel::newest()))
                .with_validity_typed(Some(bm.into())),
        ],
    )
}

fn fill_null_numeric<T>(
    ca: &ChunkedArray<T>,
    strategy: FillNullStrategy,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
{
    // Nothing to fill.
    let mut out = match strategy {
        FillNullStrategy::Min => {
            ca.fill_null_with_values(ChunkAgg::min(ca).ok_or_else(err_fill_null)?)?
        },
        FillNullStrategy::Max => {
            ca.fill_null_with_values(ChunkAgg::max(ca).ok_or_else(err_fill_null)?)?
        },
        FillNullStrategy::Mean => ca.fill_null_with_values(
            ca.mean()
                .map(|v| NumCast::from(v).unwrap())
                .ok_or_else(err_fill_null)?,
        )?,
        FillNullStrategy::One => return ca.fill_null_with_values(One::one()),
        FillNullStrategy::Zero => return ca.fill_null_with_values(Zero::zero()),
        FillNullStrategy::MinBound => return ca.fill_null_with_values(Bounded::min_value()),
        FillNullStrategy::MaxBound => return ca.fill_null_with_values(Bounded::max_value()),
        FillNullStrategy::Forward(None) => fill_forward_numeric(ca),
        FillNullStrategy::Backward(None) => fill_backward_numeric(ca),
        // Handled earlier
        FillNullStrategy::Forward(_) => unreachable!(),
        FillNullStrategy::Backward(_) => unreachable!(),
    };
    out.rename(ca.name());
    Ok(out)
}

fn fill_with_gather<F: Fn(&Bitmap) -> Vec<IdxSize>>(
    s: &Series,
    bits_to_idx: F,
) -> PolarsResult<Series> {
    let s = s.rechunk();
    let arr = s.chunks()[0].clone();
    let validity = arr.validity().expect("nulls");

    let idx = bits_to_idx(validity);

    Ok(unsafe { s.take_unchecked_from_slice(&idx) })
}

fn fill_forward_gather(s: &Series) -> PolarsResult<Series> {
    fill_with_gather(s, |validity| {
        let mut last_valid = 0;
        validity
            .iter()
            .enumerate_idx()
            .map(|(i, v)| {
                if v {
                    last_valid = i;
                    i
                } else {
                    last_valid
                }
            })
            .collect::<Vec<_>>()
    })
}

fn fill_forward_gather_limit(s: &Series, limit: IdxSize) -> PolarsResult<Series> {
    fill_with_gather(s, |validity| {
        let mut last_valid = 0;
        let mut conseq_invalid_count = 0;
        validity
            .iter()
            .enumerate_idx()
            .map(|(i, v)| {
                if v {
                    last_valid = i;
                    conseq_invalid_count = 0;
                    i
                } else if conseq_invalid_count < limit {
                    conseq_invalid_count += 1;
                    last_valid
                } else {
                    i
                }
            })
            .collect::<Vec<_>>()
    })
}

fn fill_backward_gather(s: &Series) -> PolarsResult<Series> {
    fill_with_gather(s, |validity| {
        let last = validity.len() as IdxSize - 1;
        let mut last_valid = last;
        unsafe {
            validity
                .iter()
                .rev()
                .enumerate_idx()
                .map(|(i, v)| {
                    if v {
                        last_valid = last - i;
                        last - i
                    } else {
                        last_valid
                    }
                })
                .trust_my_length((last + 1) as usize)
                .collect_reversed::<Vec<_>>()
        }
    })
}

fn fill_backward_gather_limit(s: &Series, limit: IdxSize) -> PolarsResult<Series> {
    fill_with_gather(s, |validity| {
        let last = validity.len() as IdxSize - 1;
        let mut last_valid = last;
        let mut conseq_invalid_count = 0;
        unsafe {
            validity
                .iter()
                .rev()
                .enumerate_idx()
                .map(|(i, v)| {
                    if v {
                        last_valid = last - i;
                        conseq_invalid_count = 0;
                        last - i
                    } else if conseq_invalid_count < limit {
                        conseq_invalid_count += 1;
                        last_valid
                    } else {
                        last - i
                    }
                })
                .trust_my_length((last + 1) as usize)
                .collect_reversed()
        }
    })
}

fn fill_null_bool(ca: &BooleanChunked, strategy: FillNullStrategy) -> PolarsResult<Series> {
    match strategy {
        FillNullStrategy::Min => ca
            .fill_null_with_values(ca.min().ok_or_else(err_fill_null)?)
            .map(|ca| ca.into_series()),
        FillNullStrategy::Max => ca
            .fill_null_with_values(ca.max().ok_or_else(err_fill_null)?)
            .map(|ca| ca.into_series()),
        FillNullStrategy::Mean => polars_bail!(opq = mean, "Boolean"),
        FillNullStrategy::One | FillNullStrategy::MaxBound => {
            ca.fill_null_with_values(true).map(|ca| ca.into_series())
        },
        FillNullStrategy::Zero | FillNullStrategy::MinBound => {
            ca.fill_null_with_values(false).map(|ca| ca.into_series())
        },
        FillNullStrategy::Forward(_) => unreachable!(),
        FillNullStrategy::Backward(_) => unreachable!(),
    }
}

fn fill_null_binary(ca: &BinaryChunked, strategy: FillNullStrategy) -> PolarsResult<BinaryChunked> {
    match strategy {
        FillNullStrategy::Min => {
            ca.fill_null_with_values(ca.min_binary().ok_or_else(err_fill_null)?)
        },
        FillNullStrategy::Max => {
            ca.fill_null_with_values(ca.max_binary().ok_or_else(err_fill_null)?)
        },
        FillNullStrategy::Zero => ca.fill_null_with_values(&[]),
        FillNullStrategy::Forward(_) => unreachable!(),
        FillNullStrategy::Backward(_) => unreachable!(),
        strat => polars_bail!(InvalidOperation: "fill-null strategy {:?} is not supported", strat),
    }
}

impl<T> ChunkFillNullValue<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn fill_null_with_values(&self, value: T::Native) -> PolarsResult<Self> {
        Ok(self.apply_kernel(&|arr| Box::new(set_at_nulls(arr, value))))
    }
}

impl ChunkFillNullValue<bool> for BooleanChunked {
    fn fill_null_with_values(&self, value: bool) -> PolarsResult<Self> {
        self.set(&self.is_null(), Some(value))
    }
}

impl ChunkFillNullValue<&[u8]> for BinaryChunked {
    fn fill_null_with_values(&self, value: &[u8]) -> PolarsResult<Self> {
        self.set(&self.is_null(), Some(value))
    }
}
