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

/// Perform an in-place `Iterator::filter_map` over two vectors at the same time.
pub fn inplace_zip_filtermap<T, U>(
    x: &mut Vec<T>,
    y: &mut Vec<U>,
    mut f: impl FnMut(T, U) -> Option<(T, U)>,
) {
    assert_eq!(x.len(), y.len());

    let length = x.len();

    struct OwnedBuffer<T> {
        end: *mut T,
        length: usize,
    }

    impl<T> Drop for OwnedBuffer<T> {
        fn drop(&mut self) {
            for i in 0..self.length {
                unsafe { self.end.wrapping_sub(i + 1).read() };
            }
        }
    }

    let x_ptr = x.as_mut_ptr();
    let y_ptr = y.as_mut_ptr();

    let mut x_buf = OwnedBuffer {
        end: x_ptr.wrapping_add(length),
        length,
    };
    let mut y_buf = OwnedBuffer {
        end: y_ptr.wrapping_add(length),
        length,
    };

    // SAFETY: All items are now owned by `x_buf` and `y_buf`. Since we know that `x_buf` and
    // `y_buf` will be dropped before the vecs representing `x` and `y`, this is safe.
    unsafe {
        x.set_len(0);
        y.set_len(0);
    }

    // SAFETY:
    //
    // We know we have a exclusive reference to x and y.
    //
    // We know that `i` is always smaller than `x.len()` and `y.len()`. Furthermore, we also know
    // that `i - num_deleted > 0`.
    //
    // Items are dropped exactly once, even if `f` panics.
    for i in 0..length {
        let xi = unsafe { x_ptr.wrapping_add(i).read() };
        let yi = unsafe { y_ptr.wrapping_add(i).read() };

        x_buf.length -= 1;
        y_buf.length -= 1;

        // We hold the invariant here that all items that are not yet deleted are either in
        // - `xi` or `yi`
        // - `x_buf` or `y_buf`
        // ` `x` or `y`
        //
        // This way if `f` ever panics, we are sure that all items are dropped exactly once.
        // Deleted items will be dropped when they are deleted.
        let result = f(xi, yi);

        if let Some((xi, yi)) = result {
            x.push(xi);
            y.push(yi);
        }
    }

    debug_assert_eq!(x_buf.length, 0);
    debug_assert_eq!(y_buf.length, 0);

    // We are safe to forget `x_buf` and `y_buf` here since they will not deallocate anything
    // anymore.
    std::mem::forget(x_buf);
    std::mem::forget(y_buf);
}
