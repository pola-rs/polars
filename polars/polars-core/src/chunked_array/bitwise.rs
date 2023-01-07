use std::ops::{BitAnd, BitOr, BitXor, Not};

use arrow::compute;

use super::*;
use crate::utils::{align_chunks_binary, combine_validities, CustomIterTools};

impl<T> BitAnd for &ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: BitAnd<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn bitand(self, rhs: Self) -> Self::Output {
        let (l, r) = align_chunks_binary(self, rhs);
        let chunks = l
            .downcast_iter()
            .zip(r.downcast_iter())
            .map(|(l_arr, r_arr)| {
                let l_vals = l_arr.values().as_slice();
                let r_vals = r_arr.values().as_slice();
                let validity = combine_validities(l_arr.validity(), r_arr.validity());

                let av = l_vals
                    .iter()
                    .zip(r_vals)
                    .map(|(l, r)| *l & *r)
                    .collect_trusted::<Vec<_>>();

                let arr = PrimitiveArray::new(T::get_dtype().to_arrow(), av.into(), validity);
                Box::new(arr) as ArrayRef
            })
            .collect::<Vec<_>>();

        // safety: same type
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }
}

impl<T> BitOr for &ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: BitOr<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn bitor(self, rhs: Self) -> Self::Output {
        let (l, r) = align_chunks_binary(self, rhs);
        let chunks = l
            .downcast_iter()
            .zip(r.downcast_iter())
            .map(|(l_arr, r_arr)| {
                let l_vals = l_arr.values().as_slice();
                let r_vals = r_arr.values().as_slice();
                let validity = combine_validities(l_arr.validity(), r_arr.validity());

                let av = l_vals
                    .iter()
                    .zip(r_vals)
                    .map(|(l, r)| *l | *r)
                    .collect_trusted::<Vec<_>>();

                let arr = PrimitiveArray::new(T::get_dtype().to_arrow(), av.into(), validity);
                Box::new(arr) as ArrayRef
            })
            .collect::<Vec<_>>();

        // safety: same type
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }
}

impl<T> BitXor for &ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: BitXor<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        let (l, r) = align_chunks_binary(self, rhs);
        let chunks = l
            .downcast_iter()
            .zip(r.downcast_iter())
            .map(|(l_arr, r_arr)| {
                let l_vals = l_arr.values().as_slice();
                let r_vals = r_arr.values().as_slice();
                let validity = combine_validities(l_arr.validity(), r_arr.validity());

                let av = l_vals
                    .iter()
                    .zip(r_vals)
                    .map(|(l, r)| l.bitxor(*r))
                    .collect_trusted::<Vec<_>>();

                let arr = PrimitiveArray::new(T::get_dtype().to_arrow(), av.into(), validity);
                Box::new(arr) as ArrayRef
            })
            .collect::<Vec<_>>();

        // safety: same type
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }
}

impl BitOr for &BooleanChunked {
    type Output = BooleanChunked;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self.len(), rhs.len()) {
            // make sure that we fall through if both are equal unit lengths
            // otherwise we stackoverflow
            (1, 1) => {}
            (1, _) => {
                return match self.get(0) {
                    Some(true) => BooleanChunked::full(self.name(), true, rhs.len()),
                    Some(false) => {
                        let mut rhs = rhs.clone();
                        rhs.rename(self.name());
                        rhs
                    }
                    None => &self.new_from_index(0, rhs.len()) | rhs,
                };
            }
            (_, 1) => {
                return match rhs.get(0) {
                    Some(true) => BooleanChunked::full(self.name(), true, self.len()),
                    Some(false) => self.clone(),
                    None => &rhs.new_from_index(0, self.len()) | self,
                };
            }
            _ => {}
        }

        let (lhs, rhs) = align_chunks_binary(self, rhs);
        let chunks = lhs
            .downcast_iter()
            .zip(rhs.downcast_iter())
            .map(|(lhs, rhs)| Box::new(compute::boolean_kleene::or(lhs, rhs)) as ArrayRef)
            .collect();
        // safety: same type
        unsafe { BooleanChunked::from_chunks(self.name(), chunks) }
    }
}

impl BitOr for BooleanChunked {
    type Output = BooleanChunked;

    fn bitor(self, rhs: Self) -> Self::Output {
        (&self).bitor(&rhs)
    }
}

impl BitXor for &BooleanChunked {
    type Output = BooleanChunked;

    fn bitxor(self, rhs: Self) -> Self::Output {
        match (self.len(), rhs.len()) {
            // make sure that we fall through if both are equal unit lengths
            // otherwise we stackoverflow
            (1, 1) => {}
            (1, _) => {
                return match self.get(0) {
                    Some(true) => {
                        let mut rhs = rhs.not();
                        rhs.rename(self.name());
                        rhs
                    }
                    Some(false) => {
                        let mut rhs = rhs.clone();
                        rhs.rename(self.name());
                        rhs
                    }
                    None => &self.new_from_index(0, rhs.len()) | rhs,
                };
            }
            (_, 1) => {
                return match rhs.get(0) {
                    Some(true) => self.not(),
                    Some(false) => self.clone(),
                    None => &rhs.new_from_index(0, self.len()) | self,
                };
            }
            _ => {}
        }

        let (l, r) = align_chunks_binary(self, rhs);
        let chunks = l
            .downcast_iter()
            .zip(r.downcast_iter())
            .map(|(l_arr, r_arr)| {
                let validity = combine_validities(l_arr.validity(), r_arr.validity());
                let values = l_arr.values() ^ r_arr.values();

                let arr = BooleanArray::from_data_default(values, validity);
                Box::new(arr) as ArrayRef
            })
            .collect::<Vec<_>>();

        // safety: same type
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }
}

impl BitXor for BooleanChunked {
    type Output = BooleanChunked;

    fn bitxor(self, rhs: Self) -> Self::Output {
        (&self).bitxor(&rhs)
    }
}

impl BitAnd for &BooleanChunked {
    type Output = BooleanChunked;

    fn bitand(self, rhs: Self) -> Self::Output {
        match (self.len(), rhs.len()) {
            // make sure that we fall through if both are equal unit lengths
            // otherwise we stackoverflow
            (1, 1) => {}
            (1, _) => {
                return match self.get(0) {
                    Some(true) => rhs.clone(),
                    Some(false) => BooleanChunked::full(self.name(), false, rhs.len()),
                    None => &self.new_from_index(0, rhs.len()) & rhs,
                };
            }
            (_, 1) => {
                return match rhs.get(0) {
                    Some(true) => self.clone(),
                    Some(false) => BooleanChunked::full(self.name(), false, self.len()),
                    None => self & &rhs.new_from_index(0, self.len()),
                };
            }
            _ => {}
        }

        let (lhs, rhs) = align_chunks_binary(self, rhs);
        let chunks = lhs
            .downcast_iter()
            .zip(rhs.downcast_iter())
            .map(|(lhs, rhs)| Box::new(compute::boolean_kleene::and(lhs, rhs)) as ArrayRef)
            .collect();
        // safety: same type
        unsafe { BooleanChunked::from_chunks(self.name(), chunks) }
    }
}

impl BitAnd for BooleanChunked {
    type Output = BooleanChunked;

    fn bitand(self, rhs: Self) -> Self::Output {
        (&self).bitand(&rhs)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn guard_so_issue_2494() {
        // this cause a stack overflow
        let a = BooleanChunked::new("a", [None]);
        let b = BooleanChunked::new("b", [None]);

        assert_eq!((&a).bitand(&b).null_count(), 1);
        assert_eq!((&a).bitor(&b).null_count(), 1);
        assert_eq!((&a).bitxor(&b).null_count(), 1);
    }
}
