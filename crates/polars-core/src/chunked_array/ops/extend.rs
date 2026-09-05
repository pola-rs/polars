use arrow::Either;
use polars_array::PlBooleanArrayBuilder;
use polars_array::builder::{ShareStrategy, StaticArrayBuilder};
use polars_array::concatenate::concatenate;

use crate::prelude::append::update_sorted_flag_before_append;
use crate::prelude::*;
use crate::series::IsSorted;

/// Takes the single chunk in `chunks` out, leaving no chunk behind.
///
/// The chunk is moved out of the box rather than cloned out of it, so that the array that comes
/// back holds the only reference to its buffers.
fn take_chunk<A: StaticArray + Default>(chunks: &mut Vec<PlArrayRef>) -> A {
    let mut chunk = chunks
        .pop()
        .expect("a chunked array holds at least one chunk");
    let arr = chunk
        .as_any_mut()
        .downcast_mut::<A>()
        .expect("the chunk of a typed chunked array has that type");
    std::mem::take(arr)
}

fn extend_immutable(
    immutable: &dyn PlArray,
    chunks: &mut Vec<PlArrayRef>,
    other_chunks: &[PlArrayRef],
) {
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
            self.rechunk_mut();
            return Ok(());
        }

        // Take the chunk out of `self` before reaching for a builder: whether the values
        // allocation can be reused rather than copied turns on this being the only reference
        // left to it, so the array cannot stay reachable through `self.chunks`.
        let arr = take_chunk::<PlPrimitiveArray<T::Native>>(&mut self.chunks);

        match arr.into_builder() {
            Either::Right(mut builder) => {
                // One growth for everything appended, rather than one per chunk of `other`.
                builder.reserve(other.len());
                for arr in other.downcast_iter() {
                    builder.subslice_extend(arr, 0, arr.len(), ShareStrategy::Never);
                }
                self.chunks.push(builder.freeze().into_boxed());
            },
            // The values are shared, sliced or scalar, so there is nothing to append into.
            Either::Left(immutable) => {
                extend_immutable(&immutable, &mut self.chunks, &other.chunks)
            },
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
            self.rechunk_mut();
            return Ok(());
        }

        // A boolean array holds one *bit* per element, so its values are copied into the builder
        // rather than reclaimed the way `PlPrimitiveArray::into_builder` reclaims a values buffer:
        // the copy is an eighth of a byte per element, and appending in bulk below more than pays
        // for it.
        let arr = take_chunk::<PlBooleanArray>(&mut self.chunks);

        let mut builder = PlBooleanArrayBuilder::with_capacity(arr.len() + other.len());
        builder.subslice_extend(&arr, 0, arr.len(), ShareStrategy::Never);
        for arr in other.downcast_iter() {
            builder.subslice_extend(arr, 0, arr.len(), ShareStrategy::Never);
        }
        self.chunks.push(builder.freeze().into_boxed());
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

#[cfg(feature = "dtype-categorical")]
#[doc(hidden)]
impl<T: PolarsCategoricalType> CategoricalChunked<T> {
    pub fn extend(&mut self, other: &Self) -> PolarsResult<()> {
        assert!(self.dtype() == other.dtype());
        self.phys.extend(&other.phys)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// What makes `extend` worth having over `append`: the values go into the allocation that is
    /// already there, and the result is still a single chunk.
    #[test]
    fn extend_appends_into_the_existing_allocation() {
        let values_ptr = |ca: &Int32Chunked| {
            ca.downcast_iter()
                .next()
                .unwrap()
                .flat_values()
                .unwrap()
                .as_slice()
                .as_ptr()
        };

        // Room to grow, so that appending has somewhere to go without moving what is there.
        let mut values = Vec::with_capacity(64);
        values.extend([1, 2, 3]);
        let mut ca = Int32Chunked::from_vec(PlSmallStr::from_static("a"), values);
        let before = values_ptr(&ca);

        ca.extend(&Int32Chunked::new(PlSmallStr::from_static("a"), &[4, 5]))
            .unwrap();

        assert_eq!(ca.chunks().len(), 1);
        assert_eq!(values_ptr(&ca), before);
        assert_eq!(ca.into_no_null_iter().collect::<Vec<_>>(), [1, 2, 3, 4, 5]);
    }

    #[test]
    #[allow(clippy::redundant_clone)]
    fn test_extend_primitive() -> PolarsResult<()> {
        // create a vec with overcapacity, so that we do not trigger a realloc
        // this allows us to test if the mutation was successful

        let mut values = Vec::with_capacity(32);
        values.extend_from_slice(&[1, 2, 3]);
        let mut ca = Int32Chunked::from_vec(PlSmallStr::from_static("a"), values);
        let location = ca.to_flat().cont_slice().unwrap().as_ptr() as usize;
        let to_append = Int32Chunked::new(PlSmallStr::from_static("a"), &[4, 5, 6]);

        ca.extend(&to_append)?;
        let location2 = ca.to_flat().cont_slice().unwrap().as_ptr() as usize;
        assert_eq!(location, location2);
        assert_eq!(ca.to_flat().cont_slice().unwrap(), [1, 2, 3, 4, 5, 6]);

        // now check if it succeeds if we cannot do this with a mutable.
        let _temp = ca.chunks.clone();
        ca.extend(&to_append)?;
        let location2 = ca.to_flat().cont_slice().unwrap().as_ptr() as usize;
        assert_ne!(location, location2);
        assert_eq!(
            ca.to_flat().cont_slice().unwrap(),
            [1, 2, 3, 4, 5, 6, 4, 5, 6]
        );

        Ok(())
    }

    #[test]
    fn test_extend_string() -> PolarsResult<()> {
        let mut ca = StringChunked::new(PlSmallStr::from_static("a"), &["a", "b", "c"]);
        let to_append = StringChunked::new(PlSmallStr::from_static("a"), &["a", "b", "e"]);

        ca.extend(&to_append)?;
        assert_eq!(ca.len(), 6);
        let vals = ca.no_null_iter().collect::<Vec<_>>();
        assert_eq!(vals, ["a", "b", "c", "a", "b", "e"]);

        Ok(())
    }

    #[test]
    fn test_extend_bool() -> PolarsResult<()> {
        let mut ca = BooleanChunked::new(PlSmallStr::from_static("a"), [true, false]);
        let to_append = BooleanChunked::new(PlSmallStr::from_static("a"), &[false, false]);

        ca.extend(&to_append)?;
        assert_eq!(ca.len(), 4);
        let vals = ca.no_null_iter().collect::<Vec<_>>();
        assert_eq!(vals, [true, false, false, false]);

        Ok(())
    }
}
