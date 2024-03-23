use std::mem::MaybeUninit;

use num_traits::Zero;

pub trait IntoRawParts<T> {
    fn into_raw_parts(self) -> (*mut T, usize, usize);

    // doesn't take ownership
    fn raw_parts(&self) -> (*mut T, usize, usize);
}

impl<T> IntoRawParts<T> for Vec<T> {
    fn into_raw_parts(self) -> (*mut T, usize, usize) {
        let mut me = std::mem::ManuallyDrop::new(self);
        (me.as_mut_ptr(), me.len(), me.capacity())
    }

    fn raw_parts(&self) -> (*mut T, usize, usize) {
        (self.as_ptr() as *mut T, self.len(), self.capacity())
    }
}

/// Fill current allocation if if > 0
/// otherwise realloc
pub trait ResizeFaster<T: Copy> {
    fn fill_or_alloc(&mut self, new_len: usize, value: T);
}

impl<T: Copy + Zero + PartialEq> ResizeFaster<T> for Vec<T> {
    fn fill_or_alloc(&mut self, new_len: usize, value: T) {
        if self.capacity() == 0 {
            // it is faster to allocate zeroed
            // so if the capacity is 0, we alloc (value might be 0)
            *self = vec![value; new_len]
        } else {
            // first clear then reserve so that the reserve doesn't have
            // to memcpy in case it needs to realloc.
            self.clear();
            self.reserve(new_len);

            // // init the uninit values
            let spare = &mut self.spare_capacity_mut()[..new_len];
            let init_value = MaybeUninit::new(value);
            spare.fill(init_value);
            unsafe { self.set_len(new_len) }
        }
    }
}
pub trait PushUnchecked<T> {
    /// Will push an item and not check if there is enough capacity
    ///
    /// # Safety
    /// Caller must ensure the array has enough capacity to hold `T`.
    unsafe fn push_unchecked(&mut self, value: T);
}

impl<T> PushUnchecked<T> for Vec<T> {
    #[inline]
    unsafe fn push_unchecked(&mut self, value: T) {
        debug_assert!(self.capacity() > self.len());
        let end = self.as_mut_ptr().add(self.len());
        std::ptr::write(end, value);
        self.set_len(self.len() + 1);
    }
}

pub trait CapacityByFactor {
    fn with_capacity_by_factor(original_len: usize, factor: f64) -> Self;
}

impl<T> CapacityByFactor for Vec<T> {
    fn with_capacity_by_factor(original_len: usize, factor: f64) -> Self {
        let cap = (original_len as f64 * factor) as usize;
        Vec::with_capacity(cap)
    }
}

// Trait to convert a Vec.
// The reason for this is to reduce code-generation. Conversion functions that are named
// functions should only generate the conversion loop once.
pub trait ConvertVec<Out> {
    type ItemIn;

    fn convert_owned<F: FnMut(Self::ItemIn) -> Out>(self, f: F) -> Vec<Out>;

    fn convert<F: FnMut(&Self::ItemIn) -> Out>(&self, f: F) -> Vec<Out>;
}

impl<T, Out> ConvertVec<Out> for Vec<T> {
    type ItemIn = T;

    fn convert_owned<F: FnMut(Self::ItemIn) -> Out>(self, f: F) -> Vec<Out> {
        self.into_iter().map(f).collect()
    }

    fn convert<F: FnMut(&Self::ItemIn) -> Out>(&self, f: F) -> Vec<Out> {
        self.iter().map(f).collect()
    }
}
