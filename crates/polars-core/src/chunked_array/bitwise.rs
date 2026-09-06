use std::ops::{BitAnd, BitOr, BitXor};

use polars_compute::arity::{prim_binary_values, prim_unary_values};

use super::*;
use crate::chunked_array::arity::apply_binary_kernel_broadcast_flat;

impl<T> BitAnd for &ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: BitAnd<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn bitand(self, rhs: Self) -> Self::Output {
        apply_binary_kernel_broadcast_flat(
            self,
            rhs,
            |l, r| prim_binary_values(l.clone(), r.clone(), |a, b| a & b),
            |l, r| prim_unary_values(r.clone(), |b| l & b),
            |l, r| prim_unary_values(l.clone(), |a| a & r),
        )
    }
}

impl<T> BitOr for &ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: BitOr<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn bitor(self, rhs: Self) -> Self::Output {
        apply_binary_kernel_broadcast_flat(
            self,
            rhs,
            |l, r| prim_binary_values(l.clone(), r.clone(), |a, b| a | b),
            |l, r| prim_unary_values(r.clone(), |b| l | b),
            |l, r| prim_unary_values(l.clone(), |a| a | r),
        )
    }
}

impl<T> BitXor for &ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: BitXor<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        apply_binary_kernel_broadcast_flat(
            self,
            rhs,
            |l, r| prim_binary_values(l.clone(), r.clone(), |a, b| a ^ b),
            |l, r| prim_unary_values(r.clone(), |b| l ^ b),
            |l, r| prim_unary_values(l.clone(), |a| a ^ r),
        )
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
                    Some(true) => BooleanChunked::full(self.name().clone(), true, rhs.len()),
                    Some(false) => {
                        let mut rhs = rhs.clone();
                        rhs.rename(self.name().clone());
                        rhs
                    },
                    None => &self.new_from_index(0, rhs.len()) | rhs,
                };
            },
            (_, 1) => {
                return match rhs.get(0) {
                    Some(true) => BooleanChunked::full(self.name().clone(), true, self.len()),
                    Some(false) => self.clone(),
                    None => self | &rhs.new_from_index(0, self.len()),
                };
            },
            _ => {},
        }

        arity::binary_elementwise_kernel(
            self,
            rhs,
            polars_compute::boolean::or,
            self.name().clone(),
        )
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
        if let Some((scalar, other_ca)) = match (self.len(), rhs.len()) {
            // make sure that we fall through if both are equal unit lengths
            // otherwise we stackoverflow
            (1, 1) => None,
            (1, _) => Some((self.get(0), rhs)),
            (_, 1) => Some((rhs.get(0), self)),
            _ => None,
        } {
            match scalar {
                Some(false) => other_ca.clone(),
                None => BooleanChunked::full_null(self.name().clone(), other_ca.len()),
                Some(true) => !other_ca,
            }
        } else {
            arity::binary_elementwise_kernel(
                self,
                rhs,
                polars_compute::boolean::xor,
                self.name().clone(),
            )
        }
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
                    Some(true) => rhs.clone().with_name(self.name().clone()),
                    Some(false) => BooleanChunked::full(self.name().clone(), false, rhs.len()),
                    None => &self.new_from_index(0, rhs.len()) & rhs,
                };
            },
            (_, 1) => {
                return match rhs.get(0) {
                    Some(true) => self.clone(),
                    Some(false) => BooleanChunked::full(self.name().clone(), false, self.len()),
                    None => self & &rhs.new_from_index(0, self.len()),
                };
            },
            _ => {},
        }

        arity::binary_elementwise_kernel(
            self,
            rhs,
            polars_compute::boolean::and,
            self.name().clone(),
        )
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

    /// The single chunk of `ca`, and whether its values are the one bit every element shares.
    fn values_are_repeated(ca: &BooleanChunked) -> bool {
        let [chunk] = ca.chunks().as_slice() else {
            panic!("expected a single chunk")
        };

        ca.downcast_as_array().scalar_values().is_some() && chunk.len() == ca.len()
    }

    /// A mask that is `true` or `false` throughout combines with one that is not without either
    /// of them being written out: `true` absorbs `or` and is the identity of `and`, and the other
    /// way round for `false`.
    #[test]
    fn a_repeated_bit_combines_without_being_written_out() {
        let name = PlSmallStr::from_static("a");
        let flat = BooleanChunked::new(name.clone(), [Some(true), Some(false), None]);
        let ones = BooleanChunked::full(name.clone(), true, 3);
        let zeros = BooleanChunked::full(name.clone(), false, 3);

        assert!(values_are_repeated(&ones) && values_are_repeated(&zeros));

        // The absorbing side answers for every element, in the one bit it holds.
        for absorbed in [&flat | &ones, &ones | &flat] {
            assert!(values_are_repeated(&absorbed));
            assert_eq!(absorbed.sum(), Some(3));
        }
        for absorbed in [&flat & &zeros, &zeros & &flat] {
            assert!(values_are_repeated(&absorbed));
            assert_eq!(absorbed.sum(), Some(0));
        }

        // The identity side hands the other one back as it is, nulls and all.
        for kept in [&flat | &zeros, &zeros | &flat, &flat & &ones, &ones & &flat] {
            assert_eq!(kept.len(), 3);
            assert_eq!(Vec::from(&kept), Vec::from(&flat));
        }

        // `xor` against a repeated bit leaves the other side alone or inverts it, and neither
        // writes it out; the nulls of that side carry over either way.
        assert_eq!(Vec::from(&(&flat ^ &zeros)), Vec::from(&flat));
        assert_eq!(
            Vec::from(&(&flat ^ &ones)),
            vec![Some(false), Some(true), None],
        );
    }

    #[test]
    fn guard_so_issue_2494() {
        // this cause a stack overflow
        let a = BooleanChunked::new(PlSmallStr::from_static("a"), [None]);
        let b = BooleanChunked::new(PlSmallStr::from_static("b"), [None]);

        assert_eq!((&a).bitand(&b).null_count(), 1);
        assert_eq!((&a).bitor(&b).null_count(), 1);
        assert_eq!((&a).bitxor(&b).null_count(), 1);
    }
}
