use crate::prelude::*;
use crate::utils::CustomIterTools;
use num::Bounded;
use polars_arrow::trusted_len::PushUnchecked;

pub(crate) trait JoinAsof<T: PolarsDataType> {
    fn join_asof(&self, _other: &Series) -> Result<Vec<Option<u32>>> {
        Err(PolarsError::InvalidOperation(
            format!(
                "asof join not implemented for key with dtype: {:?}",
                T::get_dtype()
            )
            .into(),
        ))
    }
}

impl<T> JoinAsof<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Bounded + PartialOrd,
{
    fn join_asof(&self, other: &Series) -> Result<Vec<Option<u32>>> {
        let other = self.unpack_series_matching_type(other)?;
        let mut rhs_iter = other.into_iter();
        let mut tuples = Vec::with_capacity(self.len());
        if self.null_count() > 0 {
            return Err(PolarsError::ComputeError(
                "keys of asof join should not have null values".into(),
            ));
        }
        if !(other.is_sorted_reverse() | other.is_sorted()) {
            eprintln!("right key of asof join is not explicitly sorted, this may lead to unexpected results");
        }

        let mut count = 0;
        let mut rhs_idx = 0;

        let mut previous_rhs_val: T::Native = Bounded::min_value();
        let mut previous_lhs_val: T::Native = Bounded::min_value();

        for arr in self.downcast_iter() {
            for &lhs_val in arr.values().as_slice() {
                if lhs_val < previous_lhs_val {
                    return Err(PolarsError::ComputeError(
                        "left key of asof join must be sorted".into(),
                    ));
                }
                if lhs_val == previous_lhs_val {
                    tuples.push(Some(rhs_idx + 1));
                    continue;
                }
                previous_lhs_val = lhs_val;

                loop {
                    match rhs_iter.next() {
                        Some(Some(rhs_val)) => {
                            if rhs_val > lhs_val {
                                if previous_rhs_val <= lhs_val && rhs_idx > 0 {
                                    tuples.push(Some(rhs_idx - 1));
                                    rhs_idx += 1;
                                    previous_rhs_val = rhs_val;
                                } else {
                                    rhs_idx += 1;
                                    previous_rhs_val = rhs_val;
                                    tuples.push(None);
                                }
                                break;
                            }
                            previous_rhs_val = rhs_val;
                            rhs_idx += 1;
                        }
                        Some(None) => {
                            rhs_idx += 1;
                        }
                        // exhausted rhs
                        None => {
                            let remaining = self.len() - count;
                            // all remaining values in left hand side
                            if previous_rhs_val < lhs_val {
                                // all remaining values in the rhs are smaller
                                // so we join with the last: the biggest
                                let iter = std::iter::repeat(Some(rhs_idx - 1))
                                    .take(remaining)
                                    .trust_my_length(remaining);
                                tuples.extend_trusted_len(iter);
                            } else {
                                // TODO: check if this branch should be removed
                                let iter = std::iter::repeat(None)
                                    .take(remaining)
                                    .trust_my_length(remaining);
                                tuples.extend_trusted_len(iter);
                            }

                            return Ok(tuples);
                        }
                    }
                }
                count += 1;
            }
        }

        Ok(tuples)
    }
}

impl JoinAsof<BooleanType> for BooleanChunked {}
impl JoinAsof<Utf8Type> for Utf8Chunked {}
impl JoinAsof<ListType> for ListChunked {}
impl JoinAsof<CategoricalType> for CategoricalChunked {}

impl DataFrame {
    /// This is similar to a left-join except that we match on nearest key rather than equal keys.
    /// The keys must be sorted to perform an asof join
    pub fn join_asof(&self, other: &DataFrame, left_on: &str, right_on: &str) -> Result<DataFrame> {
        let left_key = self.column(left_on)?;
        let right_key = other.column(right_on)?;

        let take_idx = left_key.join_asof(right_key)?;
        // Safety:
        // join tuples are in bounds
        let right_df = unsafe {
            other.take_opt_iter_unchecked(
                take_idx
                    .into_iter()
                    .map(|opt_idx| opt_idx.map(|idx| idx as usize)),
            )
        };

        self.finish_join(self.clone(), right_df, None)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::df;

    #[test]
    fn test_join_asof() -> Result<()> {
        let left = df![
            "a" => [1, 5, 10],
            "left_val" => ["a", "b", "c"]
        ]?;

        let right = df![
            "b" => [1, 2, 3, 6, 7],
            "right_val" => [1, 2, 3, 6, 7]
        ]?;

        let out = left.join_asof(&right, "a", "b")?;
        let expected = df![
            "a" => [1, 5, 10],
            "left_val" => ["a", "b", "c"],
            "b" => [1, 3, 7],
            "right_val" => [1, 3, 7]
        ]?;
        assert!(out.frame_equal_missing(&expected));

        let left = df![
            "a" => [2, 5, 10, 12],
            "left_val" => ["a", "b", "c", "d"]
        ]?;

        let right = df![
            "b" => [1, 2, 3],
            "right_val" => [1, 2, 3]
        ]?;
        let out = left.join_asof(&right, "a", "b")?;
        let expected = df![
            "a" => [2, 5, 10, 12],
            "left_val" => ["a", "b", "c", "d"],
            "b" => [Some(2), Some(3), Some(3), Some(3)],
            "right_val" => [Some(2), Some(3), Some(3), Some(3)]
        ]?;
        assert!(out.frame_equal_missing(&expected));

        let left = df![
            "a" => [-10, 5, 10],
            "left_val" => ["a", "b", "c"]
        ]?;

        let right = df![
            "b" => [1, 2, 3, 6, 7]
        ]?;

        let out = left.join_asof(&right, "a", "b")?;
        let expected = df![
            "a" => [-10, 5, 10],
            "left_val" => ["a", "b", "c"],
            "b" => [None, Some(3), Some(7)]
        ]?;
        assert!(out.frame_equal_missing(&expected));
        Ok(())
    }
}
