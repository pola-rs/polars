use crate::marked_usize::MarkedUsize;

type Pointer = *mut u8;
const MAX_INLINE_SIZE: usize = core::mem::size_of::<Pointer>();

type Array = [u8; MAX_INLINE_SIZE];
const ARRAY_DEFAULT: Array = [0; MAX_INLINE_SIZE];

union PointerOrArray {
    ptr: Pointer,
    array: Array,
}

const DATA_DEFAULT: PointerOrArray = PointerOrArray {
    array: ARRAY_DEFAULT,
};

/// `Option<Box<[u8]>>` with inlining of `<= size_of::<*mut u8>()` bytes.
pub struct SmallBytes {
    data: PointerOrArray,
    len: MarkedUsize,
}

unsafe impl Send for SmallBytes {}
unsafe impl Sync for SmallBytes {}

impl Eq for SmallBytes {}

impl SmallBytes {
    pub const NULL: SmallBytes = SmallBytes {
        data: DATA_DEFAULT,
        len: MarkedUsize::new(0, true),
    };

    pub fn from_slice(slice: &[u8]) -> Self {
        let len = slice.len();
        assert!(len <= MarkedUsize::UNMARKED_MAX);

        let data = if len <= MAX_INLINE_SIZE {
            let mut array = ARRAY_DEFAULT;
            array[..len].copy_from_slice(slice);

            PointerOrArray { array }
        } else {
            let boxed: Box<[u8]> = slice.into();
            let ptr: *mut [u8] = Box::into_raw(boxed);
            let ptr: *mut u8 = ptr.cast();

            PointerOrArray { ptr }
        };

        Self {
            data,
            len: MarkedUsize::new(len, false),
        }
    }

    pub fn from_opt_slice(slice: Option<&[u8]>) -> Self {
        if let Some(slice) = slice {
            Self::from_slice(slice)
        } else {
            Self::NULL
        }
    }

    #[inline]
    fn as_slice(&self) -> Option<&[u8]> {
        (!self.is_null()).then_some(unsafe {
            if self.is_inline() {
                self.inline_slice_unchecked()
            } else {
                self.non_inline_slice_unchecked()
            }
        })
    }

    #[inline]
    fn is_inline(&self) -> bool {
        self.len.to_usize() <= MAX_INLINE_SIZE
    }

    #[inline]
    fn is_null(&self) -> bool {
        self.len.marked()
    }

    /// # Safety
    /// `self.is_inline()`
    #[inline]
    unsafe fn inline_slice_unchecked(&self) -> &[u8] {
        unsafe { self.data.array.get_unchecked(..self.len.to_usize()) }
    }

    /// # Safety
    /// `!self.is_inline()`
    #[inline]
    unsafe fn non_inline_slice_unchecked(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.data.ptr, self.len.to_usize()) }
    }

    /// # Safety
    /// `!self.is_inline()`
    #[inline]
    unsafe fn non_inline_slice_unchecked_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.data.ptr, self.len.to_usize()) }
    }
}

impl core::fmt::Debug for SmallBytes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("SmallBytes(")?;

        if let Some(slice) = self.as_slice() {
            core::fmt::Debug::fmt(slice, f)?;
        } else {
            f.write_str("null")?;
        }

        f.write_str(")")
    }
}

impl Drop for SmallBytes {
    fn drop(&mut self) {
        if !self.is_inline() {
            unsafe {
                let ptr: *mut [u8] = self.non_inline_slice_unchecked_mut();
                let v: Box<[u8]> = Box::from_raw(ptr);
                drop(v);
            }
        }
    }
}

impl PartialEq for SmallBytes {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl core::hash::Hash for SmallBytes {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        if let Some(slice) = self.as_slice() {
            slice.hash(state)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_INLINE_SIZE, PointerOrArray, SmallBytes};
    use crate::marked_usize::MarkedUsize;

    fn hash<T: core::hash::Hash>(t: &T) -> u64 {
        let mut s = std::hash::DefaultHasher::new();
        t.hash(&mut s);
        core::hash::Hasher::finish(&s)
    }

    #[test]
    fn test_partition_key_eq() {
        // If both sides have the NULL bit set, hash / eq should all match regardless of the values
        // of any other bits.
        let lhs = SmallBytes {
            data: PointerOrArray {
                array: [0; MAX_INLINE_SIZE],
            },
            len: MarkedUsize::new(1, true),
        };

        let rhs = SmallBytes {
            data: PointerOrArray {
                array: [1; MAX_INLINE_SIZE],
            },
            len: MarkedUsize::new(2, true),
        };

        assert_eq!(lhs, rhs);
        assert_eq!(hash(&SmallBytes::NULL), hash(&rhs));
        assert_eq!(lhs, SmallBytes::NULL);
        assert_eq!(rhs, SmallBytes::NULL);

        let mut rhs = SmallBytes::from_slice(&[1; MAX_INLINE_SIZE + 1]);
        assert!(!rhs.is_null());
        rhs.len = MarkedUsize::new(rhs.len.to_usize(), true);
        assert!(rhs.is_null());

        assert_eq!(lhs, rhs);
        assert_eq!(rhs, SmallBytes::NULL);

        let lhs = SmallBytes {
            data: PointerOrArray {
                array: [0; MAX_INLINE_SIZE],
            },
            len: MarkedUsize::new(2, true),
        };

        let rhs = SmallBytes {
            data: PointerOrArray {
                array: [0; MAX_INLINE_SIZE],
            },
            len: MarkedUsize::new(2, false),
        };

        assert_ne!(lhs, rhs);
    }
}
