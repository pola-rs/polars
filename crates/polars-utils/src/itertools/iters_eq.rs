/// Returns true if both iterators have the same length, and the items at each
/// index are equal.
pub fn iters_eq<L, R, T, U>(left: L, right: R) -> bool
where
    L: IntoIterator<Item = T>,
    R: IntoIterator<Item = U>,
    T: PartialEq<U>,
    L::IntoIter: ExactSizeIterator,
    R::IntoIter: ExactSizeIterator,
{
    let left = left.into_iter();
    let right = right.into_iter();
    left.len() == right.len() && left.zip(right).all(|(l, r)| l == r)
}
