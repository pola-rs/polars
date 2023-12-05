mod enumerate_idx;
pub use enumerate_idx::EnumerateIdxTrait;

pub trait IntoIteratorCopied: IntoIterator {
    /// The type of the elements being iterated over.
    type Item;

    /// Which kind of iterator are we turning this into?
    type IntoIter: Iterator<Item = <Self as IntoIteratorCopied>::Item>;

    fn into_iter(self) -> <Self as IntoIteratorCopied>::IntoIter;
}

impl<'a, T: Copy> IntoIteratorCopied for &'a [T] {
    type Item = T;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, T>>;

    fn into_iter(self) -> <Self as IntoIteratorCopied>::IntoIter {
        self.iter().copied()
    }
}
