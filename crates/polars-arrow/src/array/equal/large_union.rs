use crate::array::{Array, LargeUnionArray};

pub(super) fn equal(lhs: &LargeUnionArray, rhs: &LargeUnionArray) -> bool {
    lhs.dtype() == rhs.dtype()
        && lhs.len() == rhs.len()
        && (0..lhs.len()).all(|i| match (lhs.is_valid(i), rhs.is_valid(i)) {
            (true, true) => {
                let (lt, lo) = lhs.index(i);
                let (rt, ro) = rhs.index(i);
                lt == rt && lhs.fields()[lt].sliced(lo, 1) == rhs.fields()[rt].sliced(ro, 1)
            },
            (l, r) => l == r,
        })
}
