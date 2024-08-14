use arrow::compute::concatenate::concatenate;
use arrow::Either;

use crate::prelude::append::update_sorted_flag_before_append;
use crate::prelude::*;
use crate::series::IsSorted;

fn extend_immutable(immutable: &dyn Array, chunks: &mut Vec<ArrayRef>, other_chunks: &[ArrayRef]) {
    let out = if chunks.len() == 1 {
        concatenate(&[immutable, &*other_chunks[0]]).unwrap()
    } else {
        let mut arrays = Vec::with_capacity(other_chunks.len() + 1);
        arrays.push(immutable);
        arrays.extend(other_chunks.iter().map(|a| &**a));
        concatenate(&arrays).unwrap()
    };

    chunks.push(out);
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    /// Extend the memory backed by this array with the values from `other`.
    ///
    /// Different from [`ChunkedArray::append`] which adds chunks to this [`ChunkedArray`] `extend`
    /// appends the data from `other` to the underlying `PrimitiveArray` and thus may cause a reallocation.
    ///
    /// However if this does not cause a reallocation, the resulting data structure will not have any extra chunks
    /// and thus will yield faster queries.
    ///
    /// Prefer `extend` over `append` when you want to do a query after a single append. For instance during
    /// online operations where you add `n` rows and rerun a query.
    ///
    /// Prefer `append` over `extend` when you want to append many times before doing a query. For instance
    /// when you read in multiple files and when to store them in a single `DataFrame`.
    /// In the latter case finish the sequence of `append` operations with a [`rechunk`](Self::rechunk).
    pub fn extend(&mut self, other: &Self) -> PolarsResult<()> {
        update_sorted_flag_before_append::<T>(self, other);
        // all to a single chunk
        if self.chunks.len() > 1 {
            self.append(other)?;
            *self = self.rechunk();
            return Ok(());
        }
        // Depending on the state of the underlying arrow array we
        // might be able to get a `MutablePrimitiveArray`
        //
        // This is only possible if the reference count of the array and its buffers are 1
        // So the logic below is needed to keep the reference count 1 if it is

        // First we must obtain an owned version of the array
        let arr = self.downcast_iter().next().unwrap();

        // increments 1
        let arr = arr.clone();

        // now we drop our owned ArrayRefs so that
        // decrements 1
        {
            self.chunks.clear();
        }

        use Either::*;

        if arr.values().is_sliced() {
            extend_immutable(&arr, &mut self.chunks, &other.chunks);
        } else {
            match arr.into_mut() {
                Left(immutable) => {
                    extend_immutable(&immutable, &mut self.chunks, &other.chunks);
                },
                Right(mut mutable) => {
                    for arr in other.downcast_iter() {
                        match arr.null_count() {
                            0 => mutable.extend_from_slice(arr.values()),
                            _ => mutable.extend_trusted_len(arr.into_iter()),
                        }
                    }
                    let arr: PrimitiveArray<T::Native> = mutable.into();
                    self.chunks.push(Box::new(arr) as ArrayRef)
                },
            }
        }
        self.compute_len();
        Ok(())
    }
}

#[doc(hidden)]
impl StringChunked {
    pub fn extend(&mut self, other: &Self) -> PolarsResult<()> {
        self.set_sorted_flag(IsSorted::Not);
        self.append(other)
    }
}

#[doc(hidden)]
impl BinaryChunked {
    pub fn extend(&mut self, other: &Self) -> PolarsResult<()> {
        self.set_sorted_flag(IsSorted::Not);
        self.append(other)
    }
}

#[doc(hidden)]
impl BinaryOffsetChunked {
    pub fn extend(&mut self, other: &Self) -> PolarsResult<()> {
        self.set_sorted_flag(IsSorted::Not);
        self.append(other)
    }
}

#[doc(hidden)]
impl BooleanChunked {
    pub fn extend(&mut self, other: &Self) -> PolarsResult<()> {
        update_sorted_flag_before_append::<BooleanType>(self, other);
        // make sure that we are a single chunk already
        if self.chunks.len() > 1 {
            self.append(other)?;
            *self = self.rechunk();
            return Ok(());
        }
        let arr = self.downcast_iter().next().unwrap();

        // increments 1
        let arr = arr.clone();

        // now we drop our owned ArrayRefs so that
        // decrements 1
        {
            self.chunks.clear();
        }

        use Either::*;

        match arr.into_mut() {
            Left(immutable) => {
                extend_immutable(&immutable, &mut self.chunks, &other.chunks);
            },
            Right(mut mutable) => {
                for arr in other.downcast_iter() {
                    mutable.extend_trusted_len(arr.into_iter())
                }
                let arr: BooleanArray = mutable.into();
                self.chunks.push(Box::new(arr) as ArrayRef)
            },
        }
        self.compute_len();
        self.set_sorted_flag(IsSorted::Not);
        Ok(())
    }
}

#[doc(hidden)]
impl ListChunked {
    pub fn extend(&mut self, other: &Self) -> PolarsResult<()> {
        // TODO! properly implement mutation
        // this is harder because we don't know the inner type of the list
        self.set_sorted_flag(IsSorted::Not);
        self.append(other)
    }
}

#[cfg(feature = "dtype-array")]
#[doc(hidden)]
impl ArrayChunked {
    pub fn extend(&mut self, other: &Self) -> PolarsResult<()> {
        // TODO! properly implement mutation
        // this is harder because we don't know the inner type of the list
        self.set_sorted_flag(IsSorted::Not);
        self.append(other)
    }
}

#[cfg(feature = "dtype-struct")]
#[doc(hidden)]
impl StructChunked {
    pub fn extend(&mut self, other: &Self) -> PolarsResult<()> {
        // TODO! properly implement mutation
        // this is harder because we don't know the inner type of the list
        self.set_sorted_flag(IsSorted::Not);
        self.append(other)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[allow(clippy::redundant_clone)]
    fn test_extend_primitive() -> PolarsResult<()> {
        // create a vec with overcapacity, so that we do not trigger a realloc
        // this allows us to test if the mutation was successful

        let mut values = Vec::with_capacity(32);
        values.extend_from_slice(&[1, 2, 3]);
        let mut ca = Int32Chunked::from_vec("a", values);
        let location = ca.cont_slice().unwrap().as_ptr() as usize;
        let to_append = Int32Chunked::new("a", &[4, 5, 6]);

        ca.extend(&to_append)?;
        let location2 = ca.cont_slice().unwrap().as_ptr() as usize;
        assert_eq!(location, location2);
        assert_eq!(ca.cont_slice().unwrap(), [1, 2, 3, 4, 5, 6]);

        // now check if it succeeds if we cannot do this with a mutable.
        let _temp = ca.chunks.clone();
        ca.extend(&to_append)?;
        let location2 = ca.cont_slice().unwrap().as_ptr() as usize;
        assert_ne!(location, location2);
        assert_eq!(ca.cont_slice().unwrap(), [1, 2, 3, 4, 5, 6, 4, 5, 6]);

        Ok(())
    }

    #[test]
    fn test_extend_string() -> PolarsResult<()> {
        let mut ca = StringChunked::new("a", &["a", "b", "c"]);
        let to_append = StringChunked::new("a", &["a", "b", "e"]);

        ca.extend(&to_append)?;
        assert_eq!(ca.len(), 6);
        let vals = ca.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(vals, ["a", "b", "c", "a", "b", "e"]);

        Ok(())
    }

    #[test]
    fn test_extend_bool() -> PolarsResult<()> {
        let mut ca = BooleanChunked::new("a", [true, false]);
        let to_append = BooleanChunked::new("a", &[false, false]);

        ca.extend(&to_append)?;
        assert_eq!(ca.len(), 4);
        let vals = ca.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(vals, [true, false, false, false]);

        Ok(())
    }
}
