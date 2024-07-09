use core::slice::SliceIndex;
use std::cmp::Ordering;
use std::mem::MaybeUninit;
use std::ops::Range;

pub trait SliceAble {
    /// # Safety
    /// no bound checks.
    unsafe fn slice_unchecked(&self, range: Range<usize>) -> Self;

    fn slice(&self, range: Range<usize>) -> Self;
}

impl<T> SliceAble for &[T] {
    unsafe fn slice_unchecked(&self, range: Range<usize>) -> Self {
        self.get_unchecked_release(range)
    }

    fn slice(&self, range: Range<usize>) -> Self {
        self.get(range).unwrap()
    }
}

pub trait Extrema<T> {
    fn min_value(&self) -> Option<&T>;
    fn max_value(&self) -> Option<&T>;
}

impl<T: PartialOrd> Extrema<T> for [T] {
    fn min_value(&self) -> Option<&T> {
        self.iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
    }

    fn max_value(&self) -> Option<&T> {
        self.iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
    }
}

pub trait SortedSlice<T> {
    fn is_sorted_ascending(&self) -> bool;
}

impl<T: PartialOrd + Copy> SortedSlice<T> for [T] {
    fn is_sorted_ascending(&self) -> bool {
        if self.is_empty() {
            true
        } else {
            let mut previous = self[0];
            let mut sorted = true;

            // don't early stop or branch
            // so it autovectorizes
            for &v in &self[1..] {
                sorted &= previous <= v;
                previous = v;
            }
            sorted
        }
    }
}

pub trait GetSaferUnchecked<T> {
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    unsafe fn get_unchecked_release<I>(&self, index: I) -> &<I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>;

    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    unsafe fn get_unchecked_release_mut<I>(
        &mut self,
        index: I,
    ) -> &mut <I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>;
}

impl<T> GetSaferUnchecked<T> for [T] {
    #[inline(always)]
    unsafe fn get_unchecked_release<I>(&self, index: I) -> &<I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>,
    {
        if cfg!(debug_assertions) {
            &self[index]
        } else {
            unsafe { self.get_unchecked(index) }
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked_release_mut<I>(
        &mut self,
        index: I,
    ) -> &mut <I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>,
    {
        if cfg!(debug_assertions) {
            &mut self[index]
        } else {
            unsafe { self.get_unchecked_mut(index) }
        }
    }
}

pub trait Slice2Uninit<T> {
    fn as_uninit(&self) -> &[MaybeUninit<T>];
}

impl<T> Slice2Uninit<T> for [T] {
    #[inline]
    fn as_uninit(&self) -> &[MaybeUninit<T>] {
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const MaybeUninit<T>, self.len()) }
    }
}

// Loads a u64 from the given byteslice, as if it were padded with zeros.
#[inline]
pub fn load_padded_le_u64(bytes: &[u8]) -> u64 {
    let len = bytes.len();
    if len >= 8 {
        return u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    }

    if len >= 4 {
        let lo = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let hi = u32::from_le_bytes(bytes[len - 4..len].try_into().unwrap());
        return (lo as u64) | ((hi as u64) << (8 * (len - 4)));
    }

    if len == 0 {
        return 0;
    }

    let lo = bytes[0] as u64;
    let mid = (bytes[len / 2] as u64) << (8 * (len / 2));
    let hi = (bytes[len - 1] as u64) << (8 * (len - 1));
    lo | mid | hi
}
