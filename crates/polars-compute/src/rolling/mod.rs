mod mean;
mod min_max;
mod moment;
pub mod no_nulls;
pub mod nulls;
pub mod quantile_filter;
mod rank;
mod sum;

mod arg_min_max;
mod min_by_max_by;
pub use min_by_max_by::*;
pub(super) mod window;
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

pub use arg_min_max::{ArgMaxWindow, ArgMinMaxWindow, ArgMinWindow};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::types::NativeType;
pub use mean::MeanWindow;
use num_traits::{Bounded, Float, NumCast, One, Zero};
use polars_array::{ArrayCollectIterExt, Flat, NoNulls, PlArray, PlPrimitiveArray};
use polars_utils::float::IsFloat;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;
pub use sum::SumWindow;
use window::*;

/// The chunk a rolling kernel reads, borrowed out of whatever representation `arr` is stored in.
///
/// A chunk whose buffers already hold one slot per element is borrowed as it stands, and a mask
/// that repeats a *set* bit is dropped rather than written out: it leaves no element null, so the
/// answer is the one the no-nulls kernels give. Only a values buffer that repeats a single value
/// is laid out, and only because the window machines walk their values as a slice — see the
/// `RollingAggWindowNoNulls::new` signature.
pub fn rolling_chunk<T: NativeType>(
    arr: &PlPrimitiveArray<T>,
) -> std::borrow::Cow<'_, Flat<PlPrimitiveArray<T>>> {
    if let (Some(values), Some(true)) = (
        arr.flat_values(),
        arr.validity().and_then(|validity| validity.scalar_value()),
    ) {
        // SAFETY: the values hold one slot per element, and dropping the mask leaves nothing else
        // that could be repeated.
        return std::borrow::Cow::Owned(unsafe {
            Flat::new(PlPrimitiveArray::from(values.clone()))
        });
    }

    arr.to_flat()
}

type Start = usize;
type End = usize;
type Idx = usize;
type WindowSize = usize;
type Len = usize;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
pub enum QuantileMethod {
    #[default]
    Nearest,
    Lower,
    Higher,
    Midpoint,
    Linear,
    Equiprobable,
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum RollingFnParams {
    Quantile(RollingQuantileParams),
    Var(RollingVarParams),
    Rank {
        method: RollingRankMethod,
        seed: Option<u64>,
    },
    Skew {
        bias: bool,
    },
    Kurtosis {
        fisher: bool,
        bias: bool,
    },
}

fn det_offsets(i: Idx, window_size: WindowSize, _len: Len) -> (usize, usize) {
    if window_size == 0 {
        return (i, i);
    }
    (i.saturating_sub(window_size - 1), i + 1)
}
fn det_offsets_center(i: Idx, window_size: WindowSize, len: Len) -> (usize, usize) {
    if window_size == 0 {
        return (i, i);
    }
    let right_window = window_size.div_ceil(2);
    (
        i.saturating_sub(window_size - right_window),
        std::cmp::min(len, i + right_window),
    )
}

fn create_validity<Fo>(
    min_periods: usize,
    len: usize,
    window_size: usize,
    det_offsets_fn: Fo,
) -> Option<MutableBitmap>
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
{
    if min_periods > 1 {
        let mut validity = MutableBitmap::with_capacity(len);
        validity.extend_constant(len, true);

        // Set the null values at the boundaries

        // Head.
        for i in 0..len {
            let (start, end) = det_offsets_fn(i, window_size, len);
            if (end - start) < min_periods {
                validity.set(i, false)
            } else {
                break;
            }
        }
        // Tail.
        for i in (0..len).rev() {
            let (start, end) = det_offsets_fn(i, window_size, len);
            if (end - start) < min_periods {
                validity.set(i, false)
            } else {
                break;
            }
        }

        Some(validity)
    } else {
        None
    }
}

// Parameters allowed for rolling operations.
#[derive(Clone, Copy, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct RollingVarParams {
    pub ddof: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct RollingQuantileParams {
    pub prob: f64,
    pub method: QuantileMethod,
}

impl Hash for RollingQuantileParams {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Will not be NaN, so hash + eq symmetry will hold.
        self.prob.to_bits().hash(state);
        self.method.hash(state);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
pub enum RollingRankMethod {
    #[default]
    Average,
    Min,
    Max,
    Dense,
    Random,
}

/// The elements of a chunk a rolling kernel handed back.
#[cfg(test)]
fn elements_of<T: NativeType>(array: &dyn PlArray) -> Vec<Option<T>> {
    array
        .as_any()
        .downcast_ref::<PlPrimitiveArray<T>>()
        .expect("the rolling kernels hand back a primitive chunk")
        .iter()
        .collect()
}

/// A chunk of the elements `values` marked by `validity`, in the flat representation the kernels
/// over a chunk with a null under it are handed.
#[cfg(test)]
fn flat_chunk<T: NativeType>(
    values: Vec<T>,
    validity: Option<polars_array::PlBitmap>,
) -> Flat<PlPrimitiveArray<T>> {
    PlPrimitiveArray::from_vec(values)
        .with_validity(validity)
        .as_flat()
        .expect("a chunk built out of a values buffer and a flat mask is flat")
        .clone()
}

#[cfg(test)]
mod rolling_chunk_tests {
    use polars_array::PlBitmap;

    use super::*;

    /// A chunk that already holds one slot per element is borrowed, not copied.
    #[test]
    fn a_flat_chunk_is_borrowed() {
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);
        assert!(matches!(rolling_chunk(&arr), std::borrow::Cow::Borrowed(_)));
    }

    /// A mask that repeats a set bit leaves no element null, so it is dropped rather than written
    /// out — and the values buffer is handed on as the very allocation it was.
    #[test]
    fn a_repeated_set_mask_is_dropped_rather_than_written_out() {
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3])
            .with_validity(Some(PlBitmap::new_scalar(true, 3)));
        let values = arr.flat_values().unwrap().as_ptr();

        let chunk = rolling_chunk(&arr);
        assert!(chunk.validity().is_none());
        assert!(chunk.as_no_nulls().is_some());
        assert_eq!(chunk.as_slice().as_ptr(), values, "the values were copied");
    }

    /// A mask that repeats an unset bit says every element is null, which the no-nulls kernels
    /// cannot be told — so it is written out for the kernels that read it.
    #[test]
    fn a_repeated_unset_mask_is_written_out() {
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3])
            .with_validity(Some(PlBitmap::new_scalar(false, 3)));

        let chunk = rolling_chunk(&arr);
        assert!(chunk.as_no_nulls().is_none());
        assert_eq!(chunk.null_count(), 3);
    }

    /// A values buffer that repeats one value is laid out, since the window machines read a slice.
    #[test]
    fn a_repeated_values_buffer_is_laid_out() {
        let arr = PlPrimitiveArray::new_scalar(7i32, 4);
        assert_eq!(rolling_chunk(&arr).as_slice(), [7, 7, 7, 7]);
    }
}
