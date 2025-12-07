/// A usize where the top bit is used as a marked indicator.
#[derive(Copy, Clone)]
pub struct MarkedUsize(usize);

impl MarkedUsize {
    pub const UNMARKED_MAX: usize = ((1 << (usize::BITS - 1)) - 1);

    #[inline(always)]
    pub fn new(idx: usize, marked: bool) -> Self {
        debug_assert!(idx >> (usize::BITS - 1) == 0);
        Self(idx | ((marked as usize) << (usize::BITS - 1)))
    }

    #[inline(always)]
    pub fn idx(&self) -> usize {
        self.0 & Self::UNMARKED_MAX
    }

    #[inline(always)]
    pub fn marked(&self) -> bool {
        (self.0 >> (usize::BITS - 1)) != 0
    }
}
