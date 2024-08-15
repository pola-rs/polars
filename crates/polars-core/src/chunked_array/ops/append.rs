use polars_error::constants::LENGTH_LIMIT_MSG;

use crate::prelude::*;
use crate::series::IsSorted;

pub(crate) fn new_chunks(chunks: &mut Vec<ArrayRef>, other: &[ArrayRef], len: usize) {
    // Replace an empty array.
    if chunks.len() == 1 && len == 0 {
        other.clone_into(chunks);
    } else {
        for chunk in other {
            if chunk.len() > 0 {
                chunks.push(chunk.clone());
            }
        }
    }
}

pub(super) fn update_sorted_flag_before_append<T>(ca: &mut ChunkedArray<T>, other: &ChunkedArray<T>)
where
    T: PolarsDataType,
    for<'a> T::Physical<'a>: TotalOrd,
{
    // Note: Do not call (first|last)_non_null on an array here before checking
    // it is sorted, otherwise it will lead to quadratic behavior.
    let sorted_flag = match (
        ca.null_count() != ca.len(),
        other.null_count() != other.len(),
    ) {
        (false, false) => IsSorted::Ascending,
        (false, true) => {
            if
            // lhs is empty, just take sorted flag from rhs
            ca.is_empty()
                || (
                    // lhs is non-empty and all-null, so rhs must have nulls ordered first
                    other.is_sorted_any() && 1 + other.last_non_null().unwrap() == other.len()
                )
            {
                other.is_sorted_flag()
            } else {
                IsSorted::Not
            }
        },
        (true, false) => {
            if
            // rhs is empty, just take sorted flag from lhs
            other.is_empty()
                || (
                    // rhs is non-empty and all-null, so lhs must have nulls ordered last
                    ca.is_sorted_any() && ca.first_non_null().unwrap() == 0
                )
            {
                ca.is_sorted_flag()
            } else {
                IsSorted::Not
            }
        },
        (true, true) => {
            // both arrays have non-null values.
            // for arrays of unit length we can ignore the sorted flag, as it is
            // not necessarily set.
            if !(ca.is_sorted_any() || ca.len() == 1)
                || !(other.is_sorted_any() || other.len() == 1)
                || !(
                    // We will coerce for single values
                    ca.len() - ca.null_count() == 1
                        || other.len() - other.null_count() == 1
                        || ca.is_sorted_flag() == other.is_sorted_flag()
                )
            {
                IsSorted::Not
            } else {
                let l_idx = ca.last_non_null().unwrap();
                let r_idx = other.first_non_null().unwrap();

                let null_pos_check =
                    // check null positions
                    // lhs does not end in nulls
                    (1 + l_idx == ca.len())
                    // rhs does not start with nulls
                    && (r_idx == 0)
                    // if there are nulls, they are all on one end
                    && !(ca.first_non_null().unwrap() != 0 && 1 + other.last_non_null().unwrap() != other.len());

                if !null_pos_check {
                    IsSorted::Not
                } else {
                    #[allow(unused_assignments)]
                    let mut out = IsSorted::Not;

                    // This can be relatively expensive because of chunks, so delay as much as possible.
                    let l_val = unsafe { ca.value_unchecked(l_idx) };
                    let r_val = unsafe { other.value_unchecked(r_idx) };

                    match (
                        ca.len() - ca.null_count() == 1,
                        other.len() - other.null_count() == 1,
                    ) {
                        (true, true) => {
                            out = [IsSorted::Descending, IsSorted::Ascending]
                                [l_val.tot_le(&r_val) as usize];
                            drop(l_val);
                            drop(r_val);
                            ca.set_sorted_flag(out);
                            return;
                        },
                        (true, false) => out = other.is_sorted_flag(),
                        _ => out = ca.is_sorted_flag(),
                    }

                    debug_assert!(!matches!(out, IsSorted::Not));

                    let check = if matches!(out, IsSorted::Ascending) {
                        l_val.tot_le(&r_val)
                    } else {
                        l_val.tot_ge(&r_val)
                    };

                    if !check {
                        out = IsSorted::Not
                    }

                    out
                }
            }
        },
    };

    ca.set_sorted_flag(sorted_flag);
}

impl<T> ChunkedArray<T>
where
    T: PolarsDataType<IsNested = FalseT>,
    for<'a> T::Physical<'a>: TotalOrd,
{
    /// Append in place. This is done by adding the chunks of `other` to this [`ChunkedArray`].
    ///
    /// See also [`extend`](Self::extend) for appends to the underlying memory
    pub fn append(&mut self, other: &Self) -> PolarsResult<()> {
        update_sorted_flag_before_append::<T>(self, other);
        let len = self.len();
        self.length = self
            .length
            .checked_add(other.length)
            .ok_or_else(|| polars_err!(ComputeError: LENGTH_LIMIT_MSG))?;
        self.null_count += other.null_count;
        new_chunks(&mut self.chunks, &other.chunks, len);
        Ok(())
    }
}

#[doc(hidden)]
impl ListChunked {
    pub fn append(&mut self, other: &Self) -> PolarsResult<()> {
        let dtype = merge_dtypes(self.dtype(), other.dtype())?;
        self.field = Arc::new(Field::new(self.name(), dtype));

        let len = self.len();
        self.length = self
            .length
            .checked_add(other.length)
            .ok_or_else(|| polars_err!(ComputeError: LENGTH_LIMIT_MSG))?;
        self.null_count += other.null_count;
        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted_flag(IsSorted::Not);
        if !other.get_fast_explode_list() {
            self.unset_fast_explode_list()
        }
        Ok(())
    }
}

#[cfg(feature = "dtype-array")]
#[doc(hidden)]
impl ArrayChunked {
    pub fn append(&mut self, other: &Self) -> PolarsResult<()> {
        let dtype = merge_dtypes(self.dtype(), other.dtype())?;
        self.field = Arc::new(Field::new(self.name(), dtype));

        let len = self.len();

        self.length = self
            .length
            .checked_add(other.length)
            .ok_or_else(|| polars_err!(ComputeError: LENGTH_LIMIT_MSG))?;
        self.null_count += other.null_count;

        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted_flag(IsSorted::Not);
        Ok(())
    }
}

#[cfg(feature = "dtype-struct")]
#[doc(hidden)]
impl StructChunked {
    pub fn append(&mut self, other: &Self) -> PolarsResult<()> {
        let dtype = merge_dtypes(self.dtype(), other.dtype())?;
        self.field = Arc::new(Field::new(self.name(), dtype));

        let len = self.len();

        self.length = self
            .length
            .checked_add(other.length)
            .ok_or_else(|| polars_err!(ComputeError: LENGTH_LIMIT_MSG))?;
        self.null_count += other.null_count;

        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted_flag(IsSorted::Not);
        Ok(())
    }
}

#[cfg(feature = "object")]
#[doc(hidden)]
impl<T: PolarsObject> ObjectChunked<T> {
    pub fn append(&mut self, other: &Self) -> PolarsResult<()> {
        let len = self.len();
        self.length = self
            .length
            .checked_add(other.length)
            .ok_or_else(|| polars_err!(ComputeError: LENGTH_LIMIT_MSG))?;
        self.null_count += other.null_count;
        self.set_sorted_flag(IsSorted::Not);
        new_chunks(&mut self.chunks, &other.chunks, len);
        Ok(())
    }
}
