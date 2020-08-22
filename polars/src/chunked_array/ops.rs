use crate::chunked_array::builder::get_large_list_builder;
/// Traits for miscellaneous operations on ChunkedArray
use crate::prelude::*;
use crate::utils::Xob;
use arrow::compute;
use itertools::Itertools;
use std::cmp::Ordering;

pub trait ChunkSort<T> {
    fn sort(&self, reverse: bool) -> ChunkedArray<T>;

    fn sort_in_place(&mut self, reverse: bool);

    fn argsort(&self, reverse: bool) -> Vec<usize>;
}

fn sort_partial<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    match (a, b) {
        (Some(a), Some(b)) => a.partial_cmp(b).expect("could not compare"),
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

impl<T> ChunkSort<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: std::cmp::PartialOrd,
{
    fn sort(&self, reverse: bool) -> ChunkedArray<T> {
        if reverse {
            self.into_iter()
                .sorted_by(|a, b| sort_partial(b, a))
                .collect()
        } else {
            self.into_iter()
                .sorted_by(|a, b| sort_partial(a, b))
                .collect()
        }
    }

    fn sort_in_place(&mut self, reverse: bool) {
        let sorted = self.sort(reverse);
        self.chunks = sorted.chunks;
    }

    fn argsort(&self, reverse: bool) -> Vec<usize> {
        if reverse {
            self.into_iter()
                .enumerate()
                .sorted_by(|(_idx_a, a), (_idx_b, b)| sort_partial(b, a))
                .map(|(idx, _v)| idx)
                .collect::<AlignedVec<usize>>()
                .0
        } else {
            self.into_iter()
                .enumerate()
                .sorted_by(|(_idx_a, a), (_idx_b, b)| sort_partial(a, b))
                .map(|(idx, _v)| idx)
                .collect::<AlignedVec<usize>>()
                .0
        }
    }
}

macro_rules! argsort {
    ($self:ident, $closure:expr) => {{
        $self
            .into_iter()
            .enumerate()
            .sorted_by($closure)
            .map(|(idx, _v)| idx)
            .collect::<AlignedVec<usize>>()
            .0
    }};
}

macro_rules! sort {
    ($self:ident, $reverse:ident) => {{
        if $reverse {
            $self.into_iter().sorted_by(|a, b| b.cmp(a)).collect()
        } else {
            $self.into_iter().sorted_by(|a, b| a.cmp(b)).collect()
        }
    }};
}

impl ChunkSort<Utf8Type> for Utf8Chunked {
    fn sort(&self, reverse: bool) -> Utf8Chunked {
        sort!(self, reverse)
    }

    fn sort_in_place(&mut self, reverse: bool) {
        let sorted = self.sort(reverse);
        self.chunks = sorted.chunks;
    }

    fn argsort(&self, reverse: bool) -> Vec<usize> {
        if reverse {
            argsort!(self, |(_idx_a, a), (_idx_b, b)| b.cmp(a))
        } else {
            argsort!(self, |(_idx_a, a), (_idx_b, b)| a.cmp(b))
        }
    }
}

impl ChunkSort<LargeListType> for LargeListChunked {
    fn sort(&self, _reverse: bool) -> Self {
        println!("A ListChunked cannot be sorted. Doing nothing");
        self.clone()
    }

    fn sort_in_place(&mut self, _reverse: bool) {
        println!("A ListChunked cannot be sorted. Doing nothing");
    }

    fn argsort(&self, _reverse: bool) -> Vec<usize> {
        println!("A ListChunked cannot be sorted. Doing nothing");
        (0..self.len()).collect()
    }
}

impl ChunkSort<BooleanType> for BooleanChunked {
    fn sort(&self, reverse: bool) -> BooleanChunked {
        sort!(self, reverse)
    }

    fn sort_in_place(&mut self, reverse: bool) {
        let sorted = self.sort(reverse);
        self.chunks = sorted.chunks;
    }

    fn argsort(&self, reverse: bool) -> Vec<usize> {
        if reverse {
            argsort!(self, |(_idx_a, a), (_idx_b, b)| b.cmp(a))
        } else {
            argsort!(self, |(_idx_a, a), (_idx_b, b)| a.cmp(b))
        }
    }
}

/// Fill a ChunkedArray with one value.
pub trait ChunkFull<T> {
    /// Create a ChunkedArray with a single value.
    fn full(name: &str, value: T, length: usize) -> Self
    where
        Self: std::marker::Sized;
}

impl<T> ChunkFull<T::Native> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    fn full(name: &str, value: T::Native, length: usize) -> Self
    where
        T::Native: Copy,
    {
        let mut builder = PrimitiveChunkedBuilder::new(name, length);

        for _ in 0..length {
            builder.append_value(value)
        }
        builder.finish()
    }
}

impl<'a> ChunkFull<&'a str> for Utf8Chunked {
    fn full(name: &str, value: &'a str, length: usize) -> Self {
        let mut builder = Utf8ChunkedBuilder::new(name, length);

        for _ in 0..length {
            builder.append_value(value);
        }
        builder.finish()
    }
}

pub trait ChunkReverse<T> {
    fn reverse(&self) -> ChunkedArray<T>;
}

impl<T> ChunkReverse<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkOps,
{
    fn reverse(&self) -> ChunkedArray<T> {
        if let Ok(slice) = self.cont_slice() {
            let ca: Xob<ChunkedArray<T>> = slice.iter().rev().copied().collect();
            let mut ca = ca.into_inner();
            ca.rename(self.name());
            ca
        } else {
            self.take((0..self.len()).rev(), None)
                .expect("implementation error, should not fail")
        }
    }
}

macro_rules! impl_reverse {
    ($arrow_type:ident, $ca_type:ident) => {
        impl ChunkReverse<$arrow_type> for $ca_type {
            fn reverse(&self) -> Self {
                self.take((0..self.len()).rev(), None)
                    .expect("implementation error, should not fail")
            }
        }
    };
}

impl_reverse!(BooleanType, BooleanChunked);
impl_reverse!(Utf8Type, Utf8Chunked);
impl_reverse!(LargeListType, LargeListChunked);

pub trait ChunkFilter<T> {
    /// Filter values in the ChunkedArray with a boolean mask.
    ///
    /// ```rust
    /// # use polars::prelude::*;
    /// let array = Int32Chunked::new_from_slice("array", &[1, 2, 3]);
    /// let mask = BooleanChunked::new_from_slice("mask", &[true, false, true]);
    ///
    /// let filtered = array.filter(&mask).unwrap();
    /// assert_eq!(Vec::from(&filtered), [Some(1), Some(3)])
    /// ```
    fn filter(&self, filter: &BooleanChunked) -> Result<ChunkedArray<T>>
    where
        Self: Sized;
}

impl<T> ChunkFilter<T> for ChunkedArray<T>
where
    T: PolarsSingleType,
    ChunkedArray<T>: ChunkOps,
{
    fn filter(&self, filter: &BooleanChunked) -> Result<ChunkedArray<T>> {
        let opt = self.optional_rechunk(filter)?;
        let left = opt.as_ref().or(Some(self)).unwrap();
        let chunks = left
            .chunks
            .iter()
            .zip(&filter.downcast_chunks())
            .map(|(arr, &fil)| compute::filter(&*(arr.clone()), fil))
            .collect::<std::result::Result<Vec<_>, arrow::error::ArrowError>>()?;

        Ok(self.copy_with_chunks(chunks))
    }
}

impl ChunkFilter<LargeListType> for LargeListChunked {
    fn filter(&self, filter: &BooleanChunked) -> Result<LargeListChunked> {
        match self.dtype() {
            ArrowDataType::LargeList(dt) => {
                let mut builder = get_large_list_builder(dt, self.len(), self.name());
                filter
                    .into_iter()
                    .zip(self.into_iter())
                    .for_each(|(opt_bool_val, opt_series)| {
                        let bool_val = opt_bool_val.unwrap_or(false);
                        let opt_val = match bool_val {
                            true => opt_series,
                            false => None,
                        };
                        builder.append_opt_series(opt_val.as_ref())
                    });
                Ok(builder.finish())
            }
            _ => panic!("should not happen"),
        }
    }
}

pub trait ChunkShift<T, V> {
    fn shift(&self, periods: i32, fill_value: Option<V>) -> Result<ChunkedArray<T>>;
}

fn chunk_shift_helper<T>(
    ca: &ChunkedArray<T>,
    builder: &mut PrimitiveChunkedBuilder<T>,
    amount: usize,
) where
    T: PolarsNumericType,
    T::Native: Copy,
{
    match ca.cont_slice() {
        // fast path
        Ok(slice) => slice
            .iter()
            .take(amount)
            .for_each(|v| builder.append_value(*v)),
        // slower path
        _ => {
            ca.into_iter()
                .take(amount)
                .for_each(|opt| builder.append_option(opt));
        }
    }
}

impl<T> ChunkShift<T, T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Copy,
{
    fn shift(&self, periods: i32, fill_value: Option<T::Native>) -> Result<ChunkedArray<T>> {
        if periods.abs() >= self.len() as i32 {
            return Err(PolarsError::OutOfBounds);
        }
        let mut builder = PrimitiveChunkedBuilder::<T>::new(self.name(), self.len());
        let amount = self.len() - periods.abs() as usize;

        // Fill the front of the array
        if periods > 0 {
            for _ in 0..periods {
                builder.append_option(fill_value)
            }
            chunk_shift_helper(self, &mut builder, amount);
        // Fill the back of the array
        } else {
            chunk_shift_helper(self, &mut builder, amount);
            for _ in 0..periods.abs() {
                builder.append_option(fill_value)
            }
        }
        Ok(builder.finish())
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_shift() {
        let ca = Int32Chunked::new_from_slice("", &[1, 2, 3]);
        let shifted = ca.shift(1, Some(0)).unwrap();
        assert_eq!(shifted.cont_slice().unwrap(), &[0, 1, 2]);
        let shifted = ca.shift(1, None).unwrap();
        assert_eq!(Vec::from(&shifted), &[None, Some(1), Some(2)]);
        let shifted = ca.shift(-1, None).unwrap();
        assert_eq!(Vec::from(&shifted), &[Some(1), Some(2), None]);
        assert!(ca.shift(3, None).is_err());
    }
}
