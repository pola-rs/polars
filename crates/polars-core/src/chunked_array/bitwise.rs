use std::ops::{BitAnd, BitOr, BitXor, Not};

use arrow::compute;
use arrow::legacy::compute::bitwise;
use arrow::legacy::utils::combine_validities_and;

use super::arithmetic::arithmetic_helper;
use super::*;

impl<T> BitAnd for &ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: BitAnd<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn bitand(self, rhs: Self) -> Self::Output {
        arithmetic_helper(self, rhs, bitwise::bitand, |a, b| a.bitand(b))
    }
}

impl<T> BitOr for &ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: BitOr<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn bitor(self, rhs: Self) -> Self::Output {
        arithmetic_helper(self, rhs, bitwise::bitor, |a, b| a.bitor(b))
    }
}

impl<T> BitXor for &ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: BitXor<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        arithmetic_helper(self, rhs, bitwise::bitxor, |a, b| a.bitxor(b))
    }
}

impl BitOr for &BooleanChunked {
    type Output = BooleanChunked;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self.len(), rhs.len()) {
            // make sure that we fall through if both are equal unit lengths
            // otherwise we stackoverflow
            (1, 1) => {},
            (1, _) => {
                return match self.get(0) {
                    Some(true) => BooleanChunked::full(self.name(), true, rhs.len()),
                    Some(false) => {
                        let mut rhs = rhs.clone();
                        rhs.rename(self.name());
                        rhs
                    },
                    None => &self.new_from_index(0, rhs.len()) | rhs,
                };
            },
            (_, 1) => {
                return match rhs.get(0) {
                    Some(true) => BooleanChunked::full(self.name(), true, self.len()),
                    Some(false) => self.clone(),
                    None => &rhs.new_from_index(0, self.len()) | self,
                };
            },
            _ => {},
        }

        arity::binary(self, rhs, compute::boolean_kleene::or)
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
            (1, 1) => {},
            (1, _) => {
                return match self.get(0) {
                    Some(true) => {
                        let mut rhs = rhs.not();
                        rhs.rename(self.name());
                        rhs
                    },
                    Some(false) => {
                        let mut rhs = rhs.clone();
                        rhs.rename(self.name());
                        rhs
                    },
                    None => &self.new_from_index(0, rhs.len()) | rhs,
                };
            },
            (_, 1) => {
                return match rhs.get(0) {
                    Some(true) => self.not(),
                    Some(false) => self.clone(),
                    None => &rhs.new_from_index(0, self.len()) | self,
                };
            },
            _ => {},
        }

        arity::binary(self, rhs, |l_arr, r_arr| {
            let validity = combine_validities_and(l_arr.validity(), r_arr.validity());
            let values = l_arr.values() ^ r_arr.values();
            BooleanArray::from_data_default(values, validity)
        })
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
            (1, 1) => {},
            (1, _) => {
                return match self.get(0) {
                    Some(true) => rhs.clone(),
                    Some(false) => BooleanChunked::full(self.name(), false, rhs.len()),
                    None => &self.new_from_index(0, rhs.len()) & rhs,
                };
            },
            (_, 1) => {
                return match rhs.get(0) {
                    Some(true) => self.clone(),
                    Some(false) => BooleanChunked::full(self.name(), false, self.len()),
                    None => self & &rhs.new_from_index(0, self.len()),
                };
            },
            _ => {},
        }

        arity::binary(self, rhs, compute::boolean_kleene::and)
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
