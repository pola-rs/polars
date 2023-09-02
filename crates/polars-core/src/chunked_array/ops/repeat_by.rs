use arrow::array::ListArray;
use polars_arrow::array::ListFromIter;

use super::RepeatBy;
use crate::prelude::*;

type LargeListArray = ListArray<i64>;

fn check_lengths(length_srs: usize, length_by: usize) -> PolarsResult<()> {
    polars_ensure!(
       (length_srs == length_by) | (length_by == 1) | (length_srs == 1),
       ComputeError: "repeat_by argument and the Series should have equal length, or at least one of them should have length 1. Series length {}, by length {}",
       length_srs, length_by
    );
    Ok(())
}

impl<T> RepeatBy for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn repeat_by(&self, by: &IdxCa) -> PolarsResult<ListChunked> {
        check_lengths(self.len(), by.len())?;

        match (self.len(), by.len()) {
            (left_len, right_len) if left_len == right_len => {
                Ok(arity::binary(self, by, |arr, by| {
                    let iter = arr.into_iter().zip(by).map(|(opt_v, opt_by)| {
                        opt_by.map(|by| std::iter::repeat(opt_v.copied()).take(*by as usize))
                    });

                    // SAFETY: length of iter is trusted.
                    unsafe {
                        LargeListArray::from_iter_primitive_trusted_len(
                            iter,
                            T::get_dtype().to_arrow(),
                        )
                    }
                }))
            },
            (_, 1) => self.repeat_by(&IdxCa::new(
                self.name(),
                std::iter::repeat(by.get(0).unwrap())
                    .take(self.len())
                    .collect::<Vec<IdxSize>>(),
            )),
            (1, _) => {
                let new_array = self.new_from_index(0, by.len());
                new_array.repeat_by(by)
            },
            // we have already checked the length
            _ => unreachable!(),
        }
    }
}

impl RepeatBy for BooleanChunked {
    fn repeat_by(&self, by: &IdxCa) -> PolarsResult<ListChunked> {
        check_lengths(self.len(), by.len())?;

        match (self.len(), by.len()) {
            (left_len, right_len) if left_len == right_len => {
                Ok(arity::binary(self, by, |arr, by| {
                    let iter = arr.into_iter().zip(by).map(|(opt_v, opt_by)| {
                        opt_by.map(|by| std::iter::repeat(opt_v).take(*by as usize))
                    });

                    // SAFETY: length of iter is trusted.
                    unsafe { LargeListArray::from_iter_bool_trusted_len(iter) }
                }))
            },
            (_, 1) => self.repeat_by(&IdxCa::new(
                self.name(),
                std::iter::repeat(by.get(0).unwrap())
                    .take(self.len())
                    .collect::<Vec<IdxSize>>(),
            )),
            (1, _) => {
                let new_array = self.new_from_index(0, by.len());
                new_array.repeat_by(by)
            },
            // we have already checked the length
            _ => unreachable!(),
        }
    }
}
impl RepeatBy for Utf8Chunked {
    fn repeat_by(&self, by: &IdxCa) -> PolarsResult<ListChunked> {
        // TODO! dispatch via binary.
        check_lengths(self.len(), by.len())?;

        match (self.len(), by.len()) {
            (left_len, right_len) if left_len == right_len => {
                Ok(arity::binary(self, by, |arr, by| {
                    let iter = arr.into_iter().zip(by).map(|(opt_v, opt_by)| {
                        opt_by.map(|by| std::iter::repeat(opt_v).take(*by as usize))
                    });

                    // SAFETY: length of iter is trusted.
                    unsafe { LargeListArray::from_iter_utf8_trusted_len(iter, self.len()) }
                }))
            },
            (_, 1) => self.repeat_by(&IdxCa::new(
                self.name(),
                std::iter::repeat(by.get(0).unwrap())
                    .take(self.len())
                    .collect::<Vec<IdxSize>>(),
            )),
            (1, _) => {
                let new_array = self.new_from_index(0, by.len());
                new_array.repeat_by(by)
            },
            // we have already checked the length
            _ => unreachable!(),
        }
    }
}

impl RepeatBy for BinaryChunked {
    fn repeat_by(&self, by: &IdxCa) -> PolarsResult<ListChunked> {
        check_lengths(self.len(), by.len())?;

        match (self.len(), by.len()) {
            (left_len, right_len) if left_len == right_len => {
                Ok(arity::binary(self, by, |arr, by| {
                    let iter = arr.into_iter().zip(by).map(|(opt_v, opt_by)| {
                        opt_by.map(|by| std::iter::repeat(opt_v).take(*by as usize))
                    });

                    // SAFETY: length of iter is trusted.
                    unsafe { LargeListArray::from_iter_binary_trusted_len(iter, self.len()) }
                }))
            },
            (_, 1) => self.repeat_by(&IdxCa::new(
                self.name(),
                std::iter::repeat(by.get(0).unwrap())
                    .take(self.len())
                    .collect::<Vec<IdxSize>>(),
            )),
            (1, _) => {
                let new_array = self.new_from_index(0, by.len());
                new_array.repeat_by(by)
            },
            // we have already checked the length
            _ => unreachable!(),
        }
    }
}
