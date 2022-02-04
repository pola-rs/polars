use crate::prelude::*;
use arrow::{compute::concatenate::concatenate, Either};

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    pub fn extend(&mut self, other: &Self) {
        // make sure that we are a single chunk already
        if self.chunks.len() > 1 {
            self.rechunk();
            self.extend(other)
        }
        // Depending on the state of the underlying arrow array we
        // might be able to get a `MutablePrimitiveArray`
        //
        // This is only possible if the reference count of the array and its buffers are 1
        // So the logic below is needed to keep the reference count 1 if it is

        // First we must obtain an owned version of the array
        let arr = self.downcast_iter().next().unwrap();

        // increments 1
        let mut arr = arr.clone();

        // now we drop our owned ArrayRefs so that
        // decrements 1
        {
            self.chunks.clear();
        }

        use Either::*;

        match arr.into_mut() {
            Left(immutable) => {
                let out = if other.chunks.len() == 1 {
                    concatenate(&[&immutable, &*other.chunks[0]]).unwrap()
                } else {
                    let mut arrays = Vec::with_capacity(other.chunks.len() + 1);
                    arrays.push(&immutable as &dyn Array);
                    arrays.extend(other.chunks.iter().map(|a| &**a));
                    concatenate(&arrays).unwrap()
                };

                self.chunks.push(Arc::from(out));
            }
            Right(mut mutable) => {
                for arr in other.downcast_iter() {
                    match arr.null_count() {
                        0 => mutable.extend_from_slice(arr.values()),
                        _ => mutable.extend_trusted_len(arr.into_iter()),
                    }
                }
                let arr: PrimitiveArray<T::Native> = mutable.into();
                self.chunks.push(Arc::new(arr) as ArrayRef)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[allow(clippy::redundant_clone)]
    fn test_extend_primitive() {
        // create a vec with overcapacity, so that we do not trigger a realloc
        // this allows us to test if the mutation was successful

        let mut values = Vec::with_capacity(32);
        values.extend_from_slice(&[1, 2, 3]);
        let mut arr = Int32Chunked::from_vec("a", values);
        let location = arr.cont_slice().unwrap().as_ptr() as usize;
        let to_append = Int32Chunked::new("a", &[4, 5, 6]);

        arr.extend(&to_append);
        let location2 = arr.cont_slice().unwrap().as_ptr() as usize;
        assert_eq!(location, location2);
        assert_eq!(arr.cont_slice().unwrap(), [1, 2, 3, 4, 5, 6]);

        // now check if it succeeds if we cannot do this with a mutable.
        let temp = arr.chunks.clone();
        arr.extend(&to_append);
        let location2 = arr.cont_slice().unwrap().as_ptr() as usize;
        assert_ne!(location, location2);
        assert_eq!(arr.cont_slice().unwrap(), [1, 2, 3, 4, 5, 6, 4, 5, 6]);
    }
}
