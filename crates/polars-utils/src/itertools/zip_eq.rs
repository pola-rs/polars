use std::cmp;

/// An iterator which iterates two other iterators simultaneously
/// and panic if they have different lengths.
#[derive(Clone, Debug)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct ZipEq<I, J> {
    a: I,
    b: J,
}

impl<I: Iterator, J: Iterator> Iterator for ZipEq<I, J> {
    type Item = (I::Item, J::Item);

    fn next(&mut self) -> Option<Self::Item> {
        match (self.a.next(), self.b.next()) {
            (None, None) => None,
            (Some(a), Some(b)) => Some((a, b)),
            (None, Some(_)) | (Some(_), None) => {
                panic!("itertools: .zip_eq() reached end of one iterator before the other")
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_lower, a_upper) = self.a.size_hint();
        let (b_lower, b_upper) = self.b.size_hint();
        let lower = cmp::min(a_lower, b_lower);
        let upper = match (a_upper, b_upper) {
            (Some(u1), Some(u2)) => Some(cmp::min(u1, u2)),
            _ => a_upper.or(b_upper),
        };
        (lower, upper)
    }
}

impl<I: ExactSizeIterator, J: ExactSizeIterator> ExactSizeIterator for ZipEq<I, J> {}

/// Zips two iterators but **panics** if they are not of the same length.
pub fn zip_eq<I, J>(i: I, j: J) -> ZipEq<I::IntoIter, J::IntoIter>
where
    I: IntoIterator,
    J: IntoIterator,
{
    ZipEq {
        a: i.into_iter(),
        b: j.into_iter(),
    }
}
