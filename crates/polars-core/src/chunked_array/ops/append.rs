use crate::prelude::*;
use crate::series::IsSorted;

pub(crate) fn new_chunks(chunks: &mut Vec<ArrayRef>, other: &[ArrayRef], len: usize) {
    // replace an empty array
    if chunks.len() == 1 && len == 0 {
        *chunks = other.to_owned();
    } else {
        chunks.extend_from_slice(other);
    }
}

pub(super) fn update_sorted_flag_before_append<'a, T>(
    ca: &mut ChunkedArray<T>,
    other: &'a ChunkedArray<T>,
) where
    T: PolarsDataType,
    &'a ChunkedArray<T>: TakeRandom,
    <&'a ChunkedArray<T> as TakeRandom>::Item: PartialOrd,
{
    let get_start_end = || {
        let end = {
            unsafe {
                // reborrow and
                // inform bchk that we still have lifetime 'a
                // this is safe as we go from &mut borrow to &
                // because the trait is only implemented for &ChunkedArray<T>
                let borrow = std::mem::transmute::<&ChunkedArray<T>, &'a ChunkedArray<T>>(ca);
                borrow.get_unchecked(ca.len() - 1)
            }
        };
        let start = unsafe { other.get_unchecked(0) };

        (start, end)
    };

    if !ca.is_empty() && !other.is_empty() {
        match (ca.is_sorted_flag(), other.is_sorted_flag()) {
            (IsSorted::Ascending, IsSorted::Ascending) => {
                let (start, end) = get_start_end();
                if end > start {
                    ca.set_sorted_flag(IsSorted::Not)
                }
            }
            (IsSorted::Descending, IsSorted::Descending) => {
                let (start, end) = get_start_end();
                if end < start {
                    ca.set_sorted_flag(IsSorted::Not)
                }
            }
            _ => ca.set_sorted_flag(IsSorted::Not),
        }
    } else if ca.is_empty() {
        ca.set_sorted_flag(other.is_sorted_flag())
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    /// Append in place. This is done by adding the chunks of `other` to this [`ChunkedArray`].
    ///
    /// See also [`extend`](Self::extend) for appends to the underlying memory
    pub fn append(&mut self, other: &Self) {
        update_sorted_flag_before_append(self, other);

        let len = self.len();
        self.length += other.length;
        new_chunks(&mut self.chunks, &other.chunks, len);
    }
}

#[doc(hidden)]
impl BooleanChunked {
    pub fn append(&mut self, other: &Self) {
        update_sorted_flag_before_append(self, other);
        let len = self.len();
        self.length += other.length;
        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted_flag(IsSorted::Not);
    }
}
#[doc(hidden)]
impl Utf8Chunked {
    pub fn append(&mut self, other: &Self) {
        update_sorted_flag_before_append(self, other);
        let len = self.len();
        self.length += other.length;
        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted_flag(IsSorted::Not);
    }
}

#[doc(hidden)]
impl BinaryChunked {
    pub fn append(&mut self, other: &Self) {
        update_sorted_flag_before_append(self, other);
        let len = self.len();
        self.length += other.length;
        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted_flag(IsSorted::Not);
    }
}

#[doc(hidden)]
impl ListChunked {
    pub fn append(&mut self, other: &Self) -> PolarsResult<()> {
        let dtype = merge_dtypes(self.dtype(), other.dtype())?;
        self.field = Arc::new(Field::new(self.name(), dtype));

        let len = self.len();
        self.length += other.length;
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
        self.set_sorted_flag(IsSorted::Not);
        new_chunks(&mut self.chunks, &other.chunks, len);
    }
}
