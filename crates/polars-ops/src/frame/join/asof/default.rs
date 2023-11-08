use arrow::array::Array;
use arrow::bitmap::Bitmap;
use num_traits::Zero;
use polars_core::prelude::*;
use polars_utils::abs_diff::AbsDiff;

use super::{
    AsofJoinBackwardState, AsofJoinForwardState, AsofJoinNearestState, AsofJoinState, AsofStrategy,
};

fn join_asof_impl<'a, T, S, F>(left: &'a T::Array, right: &'a T::Array, mut filter: F) -> IdxCa
where
    T: PolarsDataType,
    S: AsofJoinState<T::Physical<'a>>,
    F: FnMut(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    if left.len() == left.null_count() || right.len() == right.null_count() {
        return IdxCa::full_null("", left.len());
    }

    let mut out = vec![0; left.len()];
    let mut mask = vec![0; (left.len() + 7) / 8];
    let mut state = S::default();

    if left.null_count() == 0 && right.null_count() == 0 {
        for (i, val_l) in left.values_iter().enumerate() {
            if let Some(r_idx) = state.next(
                &val_l,
                // SAFETY: next() only calls with indices < right.len().
                |j| Some(unsafe { right.value_unchecked(j as usize) }),
                right.len() as IdxSize,
            ) {
                // SAFETY: r_idx is non-null and valid.
                let val_r = unsafe { right.value_unchecked(r_idx as usize) };
                out[i] = r_idx;
                mask[i / 8] |= (filter(val_l, val_r) as u8) << (i % 8);
            }
        }
    } else {
        for (i, opt_val_l) in left.iter().enumerate() {
            if let Some(val_l) = opt_val_l {
                if let Some(r_idx) = state.next(
                    &val_l,
                    // SAFETY: next() only calls with indices < right.len().
                    |j| unsafe { right.get_unchecked(j as usize) },
                    right.len() as IdxSize,
                ) {
                    // SAFETY: r_idx is non-null and valid.
                    let val_r = unsafe { right.value_unchecked(r_idx as usize) };
                    out[i] = r_idx;
                    mask[i / 8] |= (filter(val_l, val_r) as u8) << (i % 8);
                }
            }
        }
    }

    let bitmap = Bitmap::try_new(mask, out.len()).unwrap();
    IdxCa::from_vec_validity("", out, Some(bitmap))
}

fn join_asof_forward<'a, T, F>(left: &'a T::Array, right: &'a T::Array, filter: F) -> IdxCa
where
    T: PolarsDataType,
    T::Physical<'a>: PartialOrd,
    F: FnMut(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    join_asof_impl::<'a, T, AsofJoinForwardState, _>(left, right, filter)
}

fn join_asof_backward<'a, T, F>(left: &'a T::Array, right: &'a T::Array, filter: F) -> IdxCa
where
    T: PolarsDataType,
    T::Physical<'a>: PartialOrd,
    F: FnMut(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    join_asof_impl::<'a, T, AsofJoinBackwardState, _>(left, right, filter)
}

fn join_asof_nearest<'a, T, F>(left: &'a T::Array, right: &'a T::Array, filter: F) -> IdxCa
where
    T: PolarsDataType,
    T::Physical<'a>: NumericNative,
    F: FnMut(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    join_asof_impl::<'a, T, AsofJoinNearestState, _>(left, right, filter)
}

pub(crate) fn join_asof_numeric<T: PolarsNumericType>(
    input_ca: &ChunkedArray<T>,
    other: &Series,
    strategy: AsofStrategy,
    tolerance: Option<AnyValue<'static>>,
) -> PolarsResult<IdxCa> {
    let other = input_ca.unpack_series_matching_type(other)?;

    let ca = input_ca.rechunk();
    let other = other.rechunk();
    let left = ca.downcast_iter().next().unwrap();
    let right = other.downcast_iter().next().unwrap();

    let out = if let Some(t) = tolerance {
        let native_tolerance = t.extract::<T::Native>().unwrap();
        let abs_tolerance = native_tolerance.abs_diff(T::Native::zero());
        let filter = |l: T::Native, r: T::Native| l.abs_diff(r) <= abs_tolerance;
        match strategy {
            AsofStrategy::Forward => join_asof_forward::<T, _>(left, right, filter),
            AsofStrategy::Backward => join_asof_backward::<T, _>(left, right, filter),
            AsofStrategy::Nearest => join_asof_nearest::<T, _>(left, right, filter),
        }
    } else {
        let filter = |_l: T::Native, _r: T::Native| true;
        match strategy {
            AsofStrategy::Forward => join_asof_forward::<T, _>(left, right, filter),
            AsofStrategy::Backward => join_asof_backward::<T, _>(left, right, filter),
            AsofStrategy::Nearest => join_asof_nearest::<T, _>(left, right, filter),
        }
    };
    Ok(out)
}

pub(crate) fn join_asof<T>(
    input_ca: &ChunkedArray<T>,
    other: &Series,
    strategy: AsofStrategy,
) -> PolarsResult<IdxCa>
where
    T: PolarsDataType,
    for<'a> T::Physical<'a>: PartialOrd,
{
    let other = input_ca.unpack_series_matching_type(other)?;

    let ca = input_ca.rechunk();
    let other = other.rechunk();
    let left = ca.downcast_iter().next().unwrap();
    let right = other.downcast_iter().next().unwrap();

    let filter = |_l: T::Physical<'_>, _r: T::Physical<'_>| true;
    Ok(match strategy {
        AsofStrategy::Forward => join_asof_impl::<T, AsofJoinForwardState, _>(left, right, filter),
        AsofStrategy::Backward => {
            join_asof_impl::<T, AsofJoinBackwardState, _>(left, right, filter)
        },
        AsofStrategy::Nearest => unimplemented!(),
    })
}

#[cfg(test)]
mod test {
    use arrow::array::PrimitiveArray;

    use super::*;

    #[test]
    fn test_asof_backward() {
        let a = PrimitiveArray::from_slice([-1, 2, 3, 3, 3, 4]);
        let b = PrimitiveArray::from_slice([1, 2, 3, 3]);

        let tuples = join_asof_backward::<Int32Type, _>(&a, &b, |_, _| true);
        assert_eq!(tuples.len(), a.len());
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), Some(3), Some(3), Some(3), Some(3)]
        );

        let b = PrimitiveArray::from_slice([1, 2, 4, 5]);
        let tuples = join_asof_backward::<Int32Type, _>(&a, &b, |_, _| true);
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), Some(1), Some(1), Some(1), Some(2)]
        );

        let a = PrimitiveArray::from_slice([2, 4, 4, 4]);
        let b = PrimitiveArray::from_slice([1, 2, 3, 3]);
        let tuples = join_asof_backward::<Int32Type, _>(&a, &b, |_, _| true);
        assert_eq!(tuples.to_vec(), &[Some(1), Some(3), Some(3), Some(3)]);
    }

    #[test]
    fn test_asof_backward_tolerance() {
        let a = PrimitiveArray::from_slice([-1, 20, 25, 30, 30, 40]);
        let b = PrimitiveArray::from_slice([10, 20, 30, 30]);
        let tuples = join_asof_backward::<Int32Type, _>(&a, &b, |l, r| l.abs_diff(r) <= 4u32);
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), None, Some(3), Some(3), None]
        );
    }

    #[test]
    fn test_asof_forward_tolerance() {
        let a = PrimitiveArray::from_slice([-1, 20, 25, 30, 30, 40, 52]);
        let b = PrimitiveArray::from_slice([10, 20, 33, 55]);
        let tuples = join_asof_forward::<Int32Type, _>(&a, &b, |l, r| l.abs_diff(r) <= 4u32);
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), None, Some(2), Some(2), None, Some(3)]
        );
    }

    #[test]
    fn test_asof_forward() {
        let a = PrimitiveArray::from_slice([-1, 1, 2, 4, 6]);
        let b = PrimitiveArray::from_slice([1, 2, 4, 5]);

        let tuples = join_asof_forward::<Int32Type, _>(&a, &b, |_, _| true);
        assert_eq!(tuples.len(), a.len());
        assert_eq!(tuples.to_vec(), &[Some(0), Some(0), Some(1), Some(2), None]);
    }
}
