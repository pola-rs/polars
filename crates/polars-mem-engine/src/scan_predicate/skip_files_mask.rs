use arrow::bitmap::Bitmap;

#[derive(Debug, Clone)]
pub enum SkipFilesMask {
    Exclusion(Bitmap),
    Inclusion(Bitmap),
}

impl SkipFilesMask {
    pub fn len(&self) -> usize {
        match self {
            Self::Exclusion(mask) => mask.len(),
            Self::Inclusion(mask) => mask.len(),
        }
    }

    pub fn is_skipped_file(&self, index: usize) -> bool {
        match self {
            Self::Exclusion(mask) => mask.get_bit(index),
            Self::Inclusion(mask) => !mask.get_bit(index),
        }
    }

    pub fn num_skipped_files(&self) -> usize {
        match self {
            Self::Exclusion(mask) => mask.set_bits(),
            Self::Inclusion(mask) => mask.unset_bits(),
        }
    }

    pub fn trailing_skipped_files(&self) -> usize {
        match self {
            Self::Exclusion(mask) => mask.trailing_ones(),
            Self::Inclusion(mask) => mask.trailing_zeros(),
        }
    }

    pub fn sliced(self, offset: usize, len: usize) -> Self {
        match self {
            Self::Exclusion(mask) => Self::Exclusion(mask.sliced(offset, len)),
            Self::Inclusion(mask) => Self::Inclusion(mask.sliced(offset, len)),
        }
    }
}
