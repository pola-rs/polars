use std::sync::Arc;

#[derive(Clone)]
pub enum ExcludeKeysProjection {
    /// Project these indices
    Indices(Arc<[usize]>),
    /// Project this many columns from the left.
    Width(usize),
}

impl ExcludeKeysProjection {
    pub fn iter_indices(&self) -> impl ExactSizeIterator<Item = usize> {
        let (indices, end) = match self {
            Self::Indices(indices) => (indices.as_ref(), indices.len()),
            Self::Width(width) => (&[][..], *width),
        };

        (0..end).map(|i| if indices.is_empty() { i } else { indices[i] })
    }

    #[allow(unused)]
    pub fn len(&self) -> usize {
        match self {
            Self::Indices(indices) => indices.len(),
            Self::Width(len) => *len,
        }
    }
}
