mod enumerate_idx;
mod fallible;

pub use enumerate_idx::EnumerateIdxTrait;
pub use fallible::*;

pub trait IntoIteratorCopied: IntoIterator {
    /// The type of the elements being iterated over.
    type OwnedItem;

    /// Which kind of iterator are we turning this into?
    type IntoIterCopied: Iterator<Item = <Self as IntoIteratorCopied>::OwnedItem>;

    fn into_iter(self) -> <Self as IntoIteratorCopied>::IntoIterCopied;
}

impl<'a, T: Copy> IntoIteratorCopied for &'a [T] {
    type OwnedItem = T;
    type IntoIterCopied = std::iter::Copied<std::slice::Iter<'a, T>>;

    fn into_iter(self) -> <Self as IntoIteratorCopied>::IntoIterCopied {
        self.iter().copied()
    }
}
