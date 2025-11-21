pub type CatSize = u32;

pub trait CatNative {
    fn as_cat(&self) -> CatSize;
    fn from_cat(cat: CatSize) -> Self;
}

impl CatNative for u8 {
    fn as_cat(&self) -> CatSize {
        *self as CatSize
    }

    fn from_cat(cat: CatSize) -> Self {
        #[cfg(debug_assertions)]
        {
            cat.try_into().unwrap()
        }

        #[cfg(not(debug_assertions))]
        {
            cat as Self
        }
    }
}

impl CatNative for u16 {
    fn as_cat(&self) -> CatSize {
        *self as CatSize
    }

    fn from_cat(cat: CatSize) -> Self {
        #[cfg(debug_assertions)]
        {
            cat.try_into().unwrap()
        }

        #[cfg(not(debug_assertions))]
        {
            cat as Self
        }
    }
}

impl CatNative for u32 {
    fn as_cat(&self) -> CatSize {
        *self
    }

    fn from_cat(cat: CatSize) -> Self {
        cat
    }
}
