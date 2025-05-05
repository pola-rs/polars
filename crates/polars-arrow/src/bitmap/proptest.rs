use proptest::prelude::{Strategy, any_with};
use proptest::sample::SizeRange;

use super::Bitmap;

pub fn bitmap(size_range: impl Into<SizeRange>) -> impl Strategy<Value = Bitmap> {
    any_with::<Vec<bool>>(size_range.into().lift()).prop_map(Bitmap::from_iter)
}
