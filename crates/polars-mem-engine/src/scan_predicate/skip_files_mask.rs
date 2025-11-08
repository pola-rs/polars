use arrow::bitmap::Bitmap;

#[derive(Debug, Clone)]
pub enum SkipFilesMask {
    Exclusion(Bitmap),
    Inclusion(Bitmap),
}

impl SkipFilesMask {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Exclusion(mask) => mask.is_empty(),
            Self::Inclusion(mask) => mask.is_empty(),
        }
    }

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

    pub fn leading_skipped_files(&self) -> usize {
        match self {
            Self::Exclusion(mask) => mask.leading_ones(),
            Self::Inclusion(mask) => mask.leading_zeros(),
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

    pub fn non_skipped_files_idx_iter(&self) -> impl Iterator<Item = usize> + Clone {
        let range_end = self.len() - self.trailing_skipped_files();
        let range_start = if range_end == 0 {
            0
        } else {
            self.leading_skipped_files()
        };

        (range_start..range_end).filter(|i| !self.is_skipped_file(*i))
    }
}
