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
    // If either is empty, copy the sorted flag from the other.
    if ca.is_empty() {
        ca.set_sorted_flag(other.is_sorted_flag());
        return;
    }
    if other.is_empty() {
        return;
    }

    // Both need to be sorted, in the same order, if the order is maintained.
    // TODO: rework sorted flags, ascending and descending are not mutually
    // exclusive for all-equal/all-null arrays.
    let ls = ca.is_sorted_flag();
    let rs = other.is_sorted_flag();
    if ls != rs || ls == IsSorted::Not || rs == IsSorted::Not {
        ca.set_sorted_flag(IsSorted::Not);
        return;
    }

    // Check the order is maintained.
    let still_sorted = match (
        ca.null_count() != ca.len(),
        other.null_count() != other.len(),
    ) {
        (false, false) => true, // all null
        (false, true) => 1 + other.last_non_null().unwrap() == other.len(), // nulls first
        (true, false) => ca.first_non_null().unwrap() == 0, // nulls last
        (true, true) => {
            // both arrays have non-null values
            let l_idx = ca.last_non_null().unwrap();
            let r_idx = other.first_non_null().unwrap();

            let l_val = unsafe { ca.value_unchecked(l_idx) };
            let r_val = unsafe { other.value_unchecked(r_idx) };

            // compare values
            let out = if ca.is_sorted_ascending_flag() {
                l_val.tot_le(&r_val)
            } else {
                l_val.tot_ge(&r_val)
            };

            out && (
                // check null position
                // the first 2 ensure there are no nulls in the center
                (1 + l_idx == ca.len())
                && (r_idx == 0)
                // this checks that if there are nulls, they are all on one side
                && !(ca.first_non_null().unwrap() != 0 && 1 + other.last_non_null().unwrap() != other.len())
            )
        },
    };

    if !still_sorted {
        ca.set_sorted_flag(IsSorted::Not);
    }
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
