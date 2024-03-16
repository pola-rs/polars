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
            // both arrays have non-null values
            if !ca.is_sorted_any()
                || !other.is_sorted_any()
                || ca.is_sorted_flag() != other.is_sorted_flag()
            {
                IsSorted::Not
            } else {
                let l_idx = ca.last_non_null().unwrap();
                let r_idx = other.first_non_null().unwrap();

                let l_val = unsafe { ca.value_unchecked(l_idx) };
                let r_val = unsafe { other.value_unchecked(r_idx) };

                let keep_sorted =
                    // check null positions
                    // lhs does not end in nulls
                    (1 + l_idx == ca.len())
                    // rhs does not start with nulls
                    && (r_idx == 0)
                    // if there are nulls, they are all on one end
                    && !(ca.first_non_null().unwrap() != 0 && 1 + other.last_non_null().unwrap() != other.len());

                let keep_sorted = keep_sorted
                    // compare values
                    && if ca.is_sorted_ascending_flag() {
                        l_val.tot_le(&r_val)
                    } else {
                        l_val.tot_ge(&r_val)
                    };

                if keep_sorted {
                    ca.is_sorted_flag()
                } else {
                    IsSorted::Not
                }
            }
        },
    };

    ca.set_sorted_flag(sorted_flag);
}

impl<T> ChunkedArray<T>
where
    T: PolarsDataType<Structure = Flat>,
    for<'a> T::Physical<'a>: TotalOrd,
{
    /// Append in place. This is done by adding the chunks of `other` to this [`ChunkedArray`].
    ///
    /// See also [`extend`](Self::extend) for appends to the underlying memory
    pub fn append(&mut self, other: &Self) {
        update_sorted_flag_before_append::<T>(self, other);
        let len = self.len();
        self.length += other.length;
        self.null_count += other.null_count;
        new_chunks(&mut self.chunks, &other.chunks, len);
    }
}

#[doc(hidden)]
impl ListChunked {
    pub fn append(&mut self, other: &Self) -> PolarsResult<()> {
        let dtype = merge_dtypes(self.dtype(), other.dtype())?;
        self.field = Arc::new(Field::new(self.name(), dtype));

        let len = self.len();
        self.length += other.length;
        self.null_count += other.null_count;
        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted_flag(IsSorted::Not);
        if !other._can_fast_explode() {
            self.unset_fast_explode()
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
        self.length += other.length;
        self.null_count += other.null_count;
        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted_flag(IsSorted::Not);
        Ok(())
    }
}

#[cfg(feature = "object")]
#[doc(hidden)]
impl<T: PolarsObject> ObjectChunked<T> {
    pub fn append(&mut self, other: &Self) {
        let len = self.len();
        self.length += other.length;
        self.null_count += other.null_count;
        self.set_sorted_flag(IsSorted::Not);
        new_chunks(&mut self.chunks, &other.chunks, len);
    }
}
