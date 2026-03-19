use std::cmp::Ordering;

/// A pair which is ordered exclusively by the first element.
#[derive(Copy, Clone, Debug)]
pub struct Priority<P, T>(pub P, pub T);

impl<P: Ord + Eq, T> Ord for Priority<P, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<P: Ord + Eq, T> PartialOrd for Priority<P, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: Eq, T> PartialEq for Priority<P, T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<P: Eq, T> Eq for Priority<P, T> {}
