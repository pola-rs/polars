use std::fmt;
use std::sync::atomic::*;

#[derive(Default)]
#[repr(transparent)]
pub struct RelaxedCell<T: AtomicNative>(T::Atomic);

impl<T: AtomicNative> RelaxedCell<T> {
    #[inline(always)]
    pub fn load(&self) -> T {
        T::load(&self.0)
    }

    #[inline(always)]
    pub fn store(&self, value: T) {
        T::store(&self.0, value)
    }

    #[inline(always)]
    pub fn fetch_add(&self, value: T) -> T {
        T::fetch_add(&self.0, value)
    }

    #[inline(always)]
    pub fn fetch_sub(&self, value: T) -> T {
        T::fetch_sub(&self.0, value)
    }

    #[inline(always)]
    pub fn fetch_max(&self, value: T) -> T {
        T::fetch_max(&self.0, value)
    }

    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        T::get_mut(&mut self.0)
    }
}

impl<T: AtomicNative> From<T> for RelaxedCell<T> {
    #[inline(always)]
    fn from(value: T) -> Self {
        RelaxedCell(T::Atomic::from(value))
    }
}

impl<T: AtomicNative> Clone for RelaxedCell<T> {
    fn clone(&self) -> Self {
        Self(T::Atomic::from(self.load()))
    }
}

impl<T: AtomicNative> fmt::Debug for RelaxedCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RelaxedCell").field(&self.load()).finish()
    }
}

pub trait AtomicNative: Sized + Default + fmt::Debug {
    type Atomic: From<Self>;

    fn load(atomic: &Self::Atomic) -> Self;
    fn store(atomic: &Self::Atomic, val: Self);
    fn fetch_add(atomic: &Self::Atomic, val: Self) -> Self;
    fn fetch_sub(atomic: &Self::Atomic, val: Self) -> Self;
    fn fetch_max(atomic: &Self::Atomic, val: Self) -> Self;
    fn get_mut(atomic: &mut Self::Atomic) -> &mut Self;
}

macro_rules! impl_relaxed_cell {
    ($T:ty, $new:ident, $A:ty) => {
        impl RelaxedCell<$T> {
            // Not part of the trait as it should be const.
            pub const fn $new(value: $T) -> Self {
                Self(<$A>::new(value))
            }
        }

        impl AtomicNative for $T {
            type Atomic = $A;

            #[inline(always)]
            fn load(atomic: &Self::Atomic) -> Self {
                atomic.load(Ordering::Relaxed)
            }

            #[inline(always)]
            fn store(atomic: &Self::Atomic, val: Self) {
                atomic.store(val, Ordering::Relaxed);
            }

            #[inline(always)]
            fn fetch_add(atomic: &Self::Atomic, val: Self) -> Self {
                atomic.fetch_add(val, Ordering::Relaxed)
            }

            #[inline(always)]
            fn fetch_sub(atomic: &Self::Atomic, val: Self) -> Self {
                atomic.fetch_sub(val, Ordering::Relaxed)
            }

            #[inline(always)]
            fn fetch_max(atomic: &Self::Atomic, val: Self) -> Self {
                atomic.fetch_max(val, Ordering::Relaxed)
            }

            #[inline(always)]
            fn get_mut(atomic: &mut Self::Atomic) -> &mut Self {
                atomic.get_mut()
            }
        }
    };
}

impl_relaxed_cell!(u8, new_u8, AtomicU8);
impl_relaxed_cell!(u32, new_u32, AtomicU32);
impl_relaxed_cell!(u64, new_u64, AtomicU64);
impl_relaxed_cell!(usize, new_usize, AtomicUsize);

impl RelaxedCell<bool> {
    // Not part of the trait as it should be const.
    pub const fn new_bool(value: bool) -> Self {
        Self(AtomicBool::new(value))
    }

    #[inline(always)]
    pub fn fetch_or(&self, val: bool) -> bool {
        self.0.fetch_or(val, Ordering::Relaxed)
    }
}

impl AtomicNative for bool {
    type Atomic = AtomicBool;

    #[inline(always)]
    fn load(atomic: &Self::Atomic) -> Self {
        atomic.load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn store(atomic: &Self::Atomic, val: Self) {
        atomic.store(val, Ordering::Relaxed);
    }

    #[inline(always)]
    fn fetch_add(_atomic: &Self::Atomic, _val: Self) -> Self {
        unimplemented!()
    }

    #[inline(always)]
    fn fetch_sub(_atomic: &Self::Atomic, _val: Self) -> Self {
        unimplemented!()
    }

    #[inline(always)]
    fn fetch_max(atomic: &Self::Atomic, val: Self) -> Self {
        atomic.fetch_or(val, Ordering::Relaxed)
    }

    #[inline(always)]
    fn get_mut(atomic: &mut Self::Atomic) -> &mut Self {
        atomic.get_mut()
    }
}
