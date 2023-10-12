use arrow::bitmap::Bitmap;
use num_traits::{Zero, Bounded};
use polars_arrow::index::IdxSize;
use polars_core::prelude::*;
use polars_utils::abs_diff::AbsDiff;

use super::{AsofStrategy, join_asof_backward_step, join_asof_forward_step, join_asof_nearest_step};

fn join_asof_forward<T: NumericNative>(
    left: &[T],
    right: &[T],
    tolerance: T::Abs,
) -> IdxCa {
    if left.is_empty() || right.is_empty() {
        return IdxCa::full_null("", left.len());
    }

    let mut out = vec![0; left.len()];
    let mut mask = vec![0; (left.len() + 7) / 8];
    let mut offset = 0;

    for (i, &val_l) in left.iter().enumerate() {
        if join_asof_forward_step(&mut offset, val_l, |j| right[j], right.len(), tolerance) {
            out[i] = offset as IdxSize;
            mask[i / 8] |= 1 << (i % 8);
        }
    }

    let bitmap = Bitmap::try_new(mask, out.len()).unwrap();
    IdxCa::new_from_owned_with_null_bitmap("", out, Some(bitmap))
}

fn join_asof_backward<T: NumericNative>(
    left: &[T],
    right: &[T],
    tolerance: T::Abs,
) -> IdxCa {
    if left.is_empty() || right.is_empty() {
        return IdxCa::full_null("", left.len());
    }

    let mut out = vec![0; left.len()];
    let mut mask = vec![0; (left.len() + 7) / 8];
    let mut offset = right.len();

    for (i, &val_l) in left.iter().enumerate().rev() {
        if join_asof_backward_step(&mut offset, val_l, |j| right[j], tolerance) {
            out[i] = offset as IdxSize - 1;
            mask[i / 8] |= 1 << (i % 8);
        }
    }

    let bitmap = Bitmap::try_new(mask, out.len()).unwrap();
    IdxCa::new_from_owned_with_null_bitmap("", out, Some(bitmap))
}

fn join_asof_nearest<T: NumericNative>(
    left: &[T],
    right: &[T],
    tolerance: T::Abs,
) -> IdxCa {
    if left.is_empty() || right.is_empty() {
        return IdxCa::full_null("", left.len());
    }

    let mut out = vec![0; left.len()];
    let mut mask = vec![0; (left.len() + 7) / 8];
    let mut offset = right.len();

    for (i, &val_l) in left.iter().enumerate().rev() {
        if join_asof_nearest_step(&mut offset, val_l, |j| right[j], tolerance) {
            out[i] = offset as IdxSize - 1;
            mask[i / 8] |= 1 << (i % 8);
        }
    }

    let bitmap = Bitmap::try_new(mask, out.len()).unwrap();
    IdxCa::new_from_owned_with_null_bitmap("", out, Some(bitmap))
}

pub(crate) fn join_asof<T>(
    input_ca: &ChunkedArray<T>,
    other: &Series,
    strategy: AsofStrategy,
    tolerance: Option<AnyValue<'static>>,
) -> PolarsResult<IdxCa>
where
    T: PolarsNumericType,
    T::Native: Bounded + PartialOrd,
    <T::Native as AbsDiff>::Abs: Bounded,
{
    let abs_tolerance = if let Some(t) = tolerance {
        t.extract::<T::Native>().unwrap().abs_diff(T::Native::zero())
    } else {
        T::Native::max_abs_diff()       
    };
    let other = input_ca.unpack_series_matching_type(other)?;

    // Cont_slice requires a single chunk.
    let ca = input_ca.rechunk();
    let other = other.rechunk();

    let out = match strategy {
        AsofStrategy::Forward => join_asof_forward(
            ca.cont_slice().unwrap(),
            other.cont_slice().unwrap(),
            abs_tolerance,
        ),
        AsofStrategy::Backward => join_asof_backward(
            ca.cont_slice().unwrap(),
            other.cont_slice().unwrap(),
            abs_tolerance
        ),
        AsofStrategy::Nearest => join_asof_nearest(
            ca.cont_slice().unwrap(),
            other.cont_slice().unwrap(),
            abs_tolerance
        ),
    };
    Ok(out)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_asof_backward() {
        let a = [-1, 2, 3, 3, 3, 4];
        let b = [1, 2, 3, 3];

        let tuples = join_asof_backward(&a, &b, u32::MAX);
        assert_eq!(tuples.len(), a.len());
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), Some(3), Some(3), Some(3), Some(3)]
        );

        let b = [1, 2, 4, 5];
        let tuples = join_asof_backward(&a, &b, u32::MAX);
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), Some(1), Some(1), Some(1), Some(2)]
        );

        let a = [2, 4, 4, 4];
        let b = [1, 2, 3, 3];
        let tuples = join_asof_backward(&a, &b, u32::MAX);
        assert_eq!(tuples.to_vec(), &[Some(1), Some(3), Some(3), Some(3)]);
    }

    #[test]
    fn test_asof_backward_tolerance() {
        let a = [-1, 20, 25, 30, 30, 40];
        let b = [10, 20, 30, 30];
        let tuples = join_asof_backward(&a, &b, 4u32);
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), None, Some(3), Some(3), None]
        );
    }

    #[test]
    fn test_asof_forward_tolerance() {
        let a = [-1, 20, 25, 30, 30, 40, 52];
        let b = [10, 20, 33, 55];
        let tuples = join_asof_forward(&a, &b, 4u32);
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), None, Some(2), Some(2), None, Some(3)]
        );
    }

    #[test]
    fn test_asof_forward() {
        let a = [-1, 1, 2, 4, 6];
        let b = [1, 2, 4, 5];

        let tuples = join_asof_forward(&a, &b, u32::MAX);
        assert_eq!(tuples.len(), a.len());
        assert_eq!(tuples.to_vec(), &[Some(0), Some(0), Some(1), Some(2), None]);
    }
}
