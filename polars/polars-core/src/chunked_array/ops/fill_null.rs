use std::ops::Add;

use arrow::compute;
use arrow::types::simd::Simd;
use num::{Bounded, NumCast, One, Zero};
use polars_arrow::kernels::set::set_at_nulls;
use polars_arrow::trusted_len::FromIteratorReversed;
use polars_arrow::utils::{CustomIterTools, FromTrustedLenIterator};

use crate::prelude::*;

impl Series {
    /// Replace None values with one of the following strategies:
    /// * Forward fill (replace None with the previous value)
    /// * Backward fill (replace None with the next value)
    /// * Mean fill (replace None with the mean of the whole array)
    /// * Min fill (replace None with the minimum of the whole array)
    /// * Max fill (replace None with the maximum of the whole array)
    ///
    /// *NOTE: If you want to fill the Nones with a value use the
    /// [`fill_null` operation on `ChunkedArray<T>`](../chunked_array/ops/trait.ChunkFillNull.html)*.
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
    ///     Ok(())
    /// }
    /// example();
    /// ```
    pub fn fill_null(&self, strategy: FillNullStrategy) -> PolarsResult<Series> {
        let logical_type = self.dtype();
        let s = self.to_physical_repr();

        use DataType::*;
        let out = match s.dtype() {
            Boolean => fill_null_bool(s.bool().unwrap(), strategy),
            Utf8 => {
                #[cfg(feature = "dtype-binary")]
                {
                    let s = unsafe { s.cast_unchecked(&Binary)? };
                    let out = s.fill_null(strategy)?;
                    return unsafe { out.cast_unchecked(&Utf8) };
                }
                #[cfg(not(feature = "dtype-binary"))]
                {
                    panic!("activate 'dtype-binary' feature")
                }
            }
            #[cfg(feature = "dtype-binary")]
            Binary => {
                let ca = s.binary().unwrap();
                fill_null_binary(ca, strategy).map(|ca| ca.into_series())
            }
            List(_) => {
                let ca = s.list().unwrap();
                fill_null_list(ca, strategy).map(|ca| ca.into_series())
            }
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(dt, |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                        fill_null_numeric(ca, strategy).map(|ca| ca.into_series())
                })
            }
            _ => todo!(),
        }?;
        unsafe { out.cast_unchecked(logical_type) }
    }
}

// Utility trait to make generics work
trait LocalCopy {
    fn cheap_clone(&self) -> Self;
}

impl<T: Copy> LocalCopy for T {
    #[inline]
    fn cheap_clone(&self) -> Self {
        *self
    }
}

impl LocalCopy for Series {
    #[inline]
    fn cheap_clone(&self) -> Self {
        self.clone()
    }
}

fn fill_forward_limit<'a, T, K, I>(ca: &'a ChunkedArray<T>, limit: IdxSize) -> ChunkedArray<T>
where
    &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
    K: LocalCopy,
    I: TrustedLen<Item = Option<K>>,
    T: PolarsDataType,
    ChunkedArray<T>: FromTrustedLenIterator<Option<K>>,
{
    let mut cnt = 0;
    let mut previous = None;
    ca.into_iter()
        .map(|opt_v| match opt_v {
            Some(v) => {
                cnt = 0;
                previous = Some(v.cheap_clone());
                Some(v)
            }
            None => {
                if cnt < limit {
                    cnt += 1;
                    previous.as_ref().map(|v| v.cheap_clone())
                } else {
                    None
                }
            }
        })
        .collect_trusted()
}

fn fill_backward_limit<'a, T, K, I>(ca: &'a ChunkedArray<T>, limit: IdxSize) -> ChunkedArray<T>
where
    &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
    K: LocalCopy,
    I: TrustedLen<Item = Option<K>> + DoubleEndedIterator,
    T: PolarsDataType,
    ChunkedArray<T>: FromIteratorReversed<Option<K>>,
{
    let mut cnt = 0;
    let mut previous = None;
    ca.into_iter()
        .rev()
        .map(|opt_v| match opt_v {
            Some(v) => {
                cnt = 0;
                previous = Some(v.cheap_clone());
                Some(v)
            }
            None => {
                if cnt < limit {
                    cnt += 1;
                    previous.as_ref().map(|v| v.cheap_clone())
                } else {
                    None
                }
            }
        })
        .collect_reversed()
}

#[cfg(feature = "dtype-binary")]
fn fill_backward_limit_binary(ca: &BinaryChunked, limit: IdxSize) -> BinaryChunked {
    let mut cnt = 0;
    let mut previous = None;
    let out: BinaryChunked = ca
        .into_iter()
        .rev()
        .map(|opt_v| match opt_v {
            Some(v) => {
                cnt = 0;
                previous = Some(v);
                Some(v)
            }
            None => {
                if cnt < limit {
                    cnt += 1;
                    previous
                } else {
                    None
                }
            }
        })
        .collect_trusted();
    out.into_iter().rev().collect_trusted()
}

fn fill_forward<'a, T, K, I>(ca: &'a ChunkedArray<T>) -> ChunkedArray<T>
where
    &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
    K: LocalCopy,
    I: TrustedLen<Item = Option<K>>,
    T: PolarsDataType,
    ChunkedArray<T>: FromTrustedLenIterator<Option<K>>,
{
    ca.into_iter()
        .scan(None, |previous, opt_v| match opt_v {
            Some(value) => {
                *previous = Some(value.cheap_clone());
                Some(Some(value))
            }
            None => Some(previous.as_ref().map(|v| v.cheap_clone())),
        })
        .collect_trusted()
}

fn fill_backward<'a, T, K, I>(ca: &'a ChunkedArray<T>) -> ChunkedArray<T>
where
    &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
    K: Copy,
    I: TrustedLen<Item = Option<K>> + DoubleEndedIterator,
    T: PolarsDataType,
    ChunkedArray<T>: FromIteratorReversed<Option<K>>,
{
    ca.into_iter()
        .rev()
        .scan(None, |previous, opt_v| match opt_v {
            Some(value) => {
                *previous = Some(value);
                Some(Some(value))
            }
            None => Some(*previous),
        })
        .collect_reversed()
}

macro_rules! impl_fill_backward {
    ($ca:ident, $ChunkedArray:ty) => {{
        let ca: $ChunkedArray = $ca
            .into_iter()
            .rev()
            .scan(None, |previous, opt_v| match opt_v {
                Some(value) => {
                    *previous = Some(value.cheap_clone());
                    Some(Some(value))
                }
                None => Some(previous.as_ref().map(|s| s.cheap_clone())),
            })
            .collect_trusted();
        ca.into_iter().rev().collect_trusted()
    }};
}

macro_rules! impl_fill_backward_limit {
    ($ca:ident, $ChunkedArray:ty, $limit:expr) => {{
        let mut cnt = 0;
        let mut previous = None;
        let out: $ChunkedArray = $ca
            .into_iter()
            .rev()
            .map(|opt_v| match opt_v {
                Some(v) => {
                    cnt = 0;
                    previous = Some(v.cheap_clone());
                    Some(v)
                }
                None => {
                    if cnt < $limit {
                        cnt += 1;
                        previous.as_ref().map(|s| s.cheap_clone())
                    } else {
                        None
                    }
                }
            })
            .collect_trusted();
        out.into_iter().rev().collect_trusted()
    }};
}

fn fill_null_numeric<T>(
    ca: &ChunkedArray<T>,
    strategy: FillNullStrategy,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsNumericType,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
{
    // nothing to fill
    if !ca.has_validity() {
        return Ok(ca.clone());
    }
    let mut out =
        match strategy {
            FillNullStrategy::Forward(None) => fill_forward(ca),
            FillNullStrategy::Forward(Some(limit)) => fill_forward_limit(ca, limit),
            FillNullStrategy::Backward(None) => fill_backward(ca),
            FillNullStrategy::Backward(Some(limit)) => fill_backward_limit(ca, limit),
            FillNullStrategy::Min => ca.fill_null_with_values(ca.min().ok_or_else(|| {
                PolarsError::ComputeError("Could not determine fill value".into())
            })?)?,
            FillNullStrategy::Max => ca.fill_null_with_values(ca.max().ok_or_else(|| {
                PolarsError::ComputeError("Could not determine fill value".into())
            })?)?,
            FillNullStrategy::Mean => {
                ca.fill_null_with_values(ca.mean().map(|v| NumCast::from(v).unwrap()).ok_or_else(
                    || PolarsError::ComputeError("Could not determine fill value".into()),
                )?)?
            }
            FillNullStrategy::One => return ca.fill_null_with_values(One::one()),
            FillNullStrategy::Zero => return ca.fill_null_with_values(Zero::zero()),
            FillNullStrategy::MinBound => return ca.fill_null_with_values(Bounded::min_value()),
            FillNullStrategy::MaxBound => return ca.fill_null_with_values(Bounded::max_value()),
        };
    out.rename(ca.name());
    Ok(out)
}

fn fill_null_bool(ca: &BooleanChunked, strategy: FillNullStrategy) -> PolarsResult<Series> {
    // nothing to fill
    if !ca.has_validity() {
        return Ok(ca.clone().into_series());
    }
    match strategy {
        FillNullStrategy::Forward(limit) => {
            let mut out: BooleanChunked = match limit {
                Some(limit) => fill_forward_limit(ca, limit),
                None => fill_forward(ca),
            };
            out.rename(ca.name());
            Ok(out.into_series())
        }
        FillNullStrategy::Backward(limit) => {
            let mut out: BooleanChunked = match limit {
                None => fill_backward(ca),
                Some(limit) => fill_backward_limit(ca, limit),
            };
            out.rename(ca.name());
            Ok(out.into_series())
        }
        FillNullStrategy::Min => ca
            .fill_null_with_values(
                1 == ca.min().ok_or_else(|| {
                    PolarsError::ComputeError("Could not determine fill value".into())
                })?,
            )
            .map(|ca| ca.into_series()),
        FillNullStrategy::Max => ca
            .fill_null_with_values(
                1 == ca.max().ok_or_else(|| {
                    PolarsError::ComputeError("Could not determine fill value".into())
                })?,
            )
            .map(|ca| ca.into_series()),
        FillNullStrategy::Mean => Err(PolarsError::InvalidOperation(
            "mean not supported on array of Boolean type".into(),
        )),
        FillNullStrategy::One | FillNullStrategy::MaxBound => {
            ca.fill_null_with_values(true).map(|ca| ca.into_series())
        }
        FillNullStrategy::Zero | FillNullStrategy::MinBound => {
            ca.fill_null_with_values(false).map(|ca| ca.into_series())
        }
    }
}

#[cfg(feature = "dtype-binary")]
fn fill_null_binary(ca: &BinaryChunked, strategy: FillNullStrategy) -> PolarsResult<BinaryChunked> {
    // nothing to fill
    if !ca.has_validity() {
        return Ok(ca.clone());
    }
    match strategy {
        FillNullStrategy::Forward(limit) => {
            let mut out: BinaryChunked = match limit {
                Some(limit) => fill_forward_limit(ca, limit),
                None => fill_forward(ca),
            };
            out.rename(ca.name());
            Ok(out)
        }
        FillNullStrategy::Backward(limit) => {
            let mut out = match limit {
                None => impl_fill_backward!(ca, BinaryChunked),
                Some(limit) => fill_backward_limit_binary(ca, limit),
            };
            out.rename(ca.name());
            Ok(out)
        }
        strat => Err(PolarsError::InvalidOperation(
            format!("Strategy {strat:?} not supported").into(),
        )),
    }
}

fn fill_null_list(ca: &ListChunked, strategy: FillNullStrategy) -> PolarsResult<ListChunked> {
    // nothing to fill
    if !ca.has_validity() {
        return Ok(ca.clone());
    }
    match strategy {
        FillNullStrategy::Forward(limit) => {
            let mut out: ListChunked = match limit {
                Some(limit) => fill_forward_limit(ca, limit),
                None => fill_forward(ca),
            };
            out.rename(ca.name());
            Ok(out)
        }
        FillNullStrategy::Backward(limit) => {
            let mut out: ListChunked = match limit {
                None => impl_fill_backward!(ca, ListChunked),
                Some(limit) => impl_fill_backward_limit!(ca, ListChunked, limit),
            };
            out.rename(ca.name());
            Ok(out)
        }
        strat => Err(PolarsError::InvalidOperation(
            format!("Strategy {strat:?} not supported").into(),
        )),
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

#[cfg(feature = "dtype-binary")]
impl ChunkFillNullValue<&[u8]> for BinaryChunked {
    fn fill_null_with_values(&self, value: &[u8]) -> PolarsResult<Self> {
        self.set(&self.is_null(), Some(value))
    }
}
