use std::io;
use std::sync::Arc;

pub use memmap::Mmap;

mod private {
    use std::ops::Deref;
    use std::sync::Arc;

    pub use memmap::Mmap;

    use crate::mem::prefetch_l2;

    /// A read-only reference to a slice of memory that can potentially be memory-mapped.
    ///
    /// A reference count is kept to the underlying buffer to ensure the memory is kept alive.
    /// [`MemSlice::slice`] can be used to slice the memory in a zero-copy manner.
    ///
    /// This still owns the all the original memory and therefore should probably not be a long-lasting
    /// structure.
    #[derive(Clone, Debug)]
    pub struct MemSlice {
        // Store the `&[u8]` to make the `Deref` free.
        // `slice` is not 'static - it is backed by `inner`. This is safe as long as `slice` is not
        // directly accessed, and we are in a private module to guarantee that. Access should only
        // be done through `Deref<Target = [u8]>`, which automatically gives the correct lifetime.
        slice: &'static [u8],
        #[allow(unused)]
        inner: MemSliceInner,
    }

    /// Keeps the underlying buffer alive. This should be cheaply cloneable.
    #[derive(Clone, Debug)]
    #[allow(unused)]
    enum MemSliceInner {
        Bytes(bytes::Bytes),
        Mmap(Arc<Mmap>),
    }

    impl Deref for MemSlice {
        type Target = [u8];

        #[inline(always)]
        fn deref(&self) -> &Self::Target {
            self.slice
        }
    }

    impl Default for MemSlice {
        fn default() -> Self {
            Self::from_bytes(bytes::Bytes::new())
        }
    }

    impl MemSlice {
        /// Copy the contents into a new owned `Vec`
        #[inline(always)]
        pub fn to_vec(self) -> Vec<u8> {
            <[u8]>::to_vec(self.deref())
        }

        /// Construct a `MemSlice` from an existing `Vec<u8>`. This is zero-copy.
        #[inline]
        pub fn from_vec(v: Vec<u8>) -> Self {
            Self::from_bytes(bytes::Bytes::from(v))
        }

        /// Construct a `MemSlice` from [`bytes::Bytes`]. This is zero-copy.
        #[inline]
        pub fn from_bytes(bytes: bytes::Bytes) -> Self {
            Self {
                slice: unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes.as_ref()) },
                inner: MemSliceInner::Bytes(bytes),
            }
        }

        #[inline]
        pub fn from_mmap(mmap: Arc<Mmap>) -> Self {
            Self {
                slice: unsafe {
                    std::mem::transmute::<&[u8], &'static [u8]>(mmap.as_ref().as_ref())
                },
                inner: MemSliceInner::Mmap(mmap),
            }
        }

        /// Construct a `MemSlice` that simply wraps around a `&[u8]`.
        #[inline]
        pub fn from_slice(slice: &'static [u8]) -> Self {
            Self::from_bytes(bytes::Bytes::from_static(slice))
        }

        /// Attempt to prefetch the memory belonging to to this [`MemSlice`]
        #[inline]
        pub fn prefetch(&self) {
            prefetch_l2(self.as_ref());
        }

        /// # Panics
        /// Panics if range is not in bounds.
        #[inline]
        #[track_caller]
        pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
            let mut out = self.clone();
            out.slice = &out.slice[range];
            out
        }
    }
}

pub use private::MemSlice;

/// A cursor over a [`MemSlice`].
#[derive(Debug, Clone)]
pub struct MemReader {
    data: MemSlice,
    position: usize,
}

impl MemReader {
    pub fn new(data: MemSlice) -> Self {
        Self { data, position: 0 }
    }

    #[inline(always)]
    pub fn remaining_len(&self) -> usize {
        self.data.len() - self.position
    }

    #[inline(always)]
    pub fn total_len(&self) -> usize {
        self.data.len()
    }

    #[inline(always)]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Construct a `MemSlice` from an existing `Vec<u8>`. This is zero-copy.
    #[inline(always)]
    pub fn from_vec(v: Vec<u8>) -> Self {
        Self::new(MemSlice::from_vec(v))
    }

    /// Construct a `MemSlice` from [`bytes::Bytes`]. This is zero-copy.
    #[inline(always)]
    pub fn from_bytes(bytes: bytes::Bytes) -> Self {
        Self::new(MemSlice::from_bytes(bytes))
    }

    #[inline(always)]
    pub fn from_mmap(mmap: Arc<Mmap>) -> Self {
        Self::new(MemSlice::from_mmap(mmap))
    }

    // Construct a `MemSlice` that simply wraps around a `&[u8]`. The caller must ensure the
    /// slice outlives the returned `MemSlice`.
    #[inline]
    pub fn from_slice(slice: &'static [u8]) -> Self {
        Self::new(MemSlice::from_slice(slice))
    }

    #[inline(always)]
    pub fn from_reader<R: io::Read>(mut reader: R) -> io::Result<Self> {
        let mut vec = Vec::new();
        reader.read_to_end(&mut vec)?;
        Ok(Self::from_vec(vec))
    }

    #[inline(always)]
    pub fn read_slice(&mut self, n: usize) -> MemSlice {
        let start = self.position;
        let end = usize::min(self.position + n, self.data.len());
        self.position = end;
        self.data.slice(start..end)
    }
}

impl From<MemSlice> for MemReader {
    fn from(data: MemSlice) -> Self {
        Self { data, position: 0 }
    }
}

impl io::Read for MemReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = usize::min(buf.len(), self.remaining_len());
        buf[..n].copy_from_slice(&self.data[self.position..self.position + n]);
        self.position += n;
        Ok(n)
    }
}

impl io::Seek for MemReader {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        let position = match pos {
            io::SeekFrom::Start(position) => usize::min(position as usize, self.total_len()),
            io::SeekFrom::End(offset) => {
                let Some(position) = self.total_len().checked_add_signed(offset as isize) else {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "Seek before to before buffer",
                    ));
                };

                position
            },
            io::SeekFrom::Current(offset) => {
                let Some(position) = self.position.checked_add_signed(offset as isize) else {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "Seek before to before buffer",
                    ));
                };

                position
            },
        };

        self.position = position;

        Ok(position as u64)
    }
}

mod tests {

    #[test]
    fn test_mem_slice_zero_copy() {
        use std::sync::Arc;

        use super::MemSlice;

        {
            let vec = vec![1u8, 2, 3, 4, 5];
            let ptr = vec.as_ptr();

            let mem_slice = MemSlice::from_vec(vec);
            let ptr_out = mem_slice.as_ptr();

            assert_eq!(ptr_out, ptr);
        }

        {
            let mut vec = vec![1u8, 2, 3, 4, 5];
            vec.truncate(2);
            let ptr = vec.as_ptr();

            let mem_slice = MemSlice::from_vec(vec);
            let ptr_out = mem_slice.as_ptr();

            assert_eq!(ptr_out, ptr);
        }

        {
            let bytes = bytes::Bytes::from(vec![1u8, 2, 3, 4, 5]);
            let ptr = bytes.as_ptr();

            let mem_slice = MemSlice::from_bytes(bytes);
            let ptr_out = mem_slice.as_ptr();

            assert_eq!(ptr_out, ptr);
        }

        {
            let path = "../../examples/datasets/foods1.csv";
            let file = std::fs::File::open(path).unwrap();
            let mmap = unsafe { memmap::Mmap::map(&file) }.unwrap();
            let ptr = mmap.as_ptr();

            let mem_slice = MemSlice::from_mmap(Arc::new(mmap));
            let ptr_out = mem_slice.as_ptr();

            assert_eq!(ptr_out, ptr);
        }

        {
            let vec = vec![1u8, 2, 3, 4, 5];
            let slice = vec.as_slice();
            let ptr = slice.as_ptr();

            let mem_slice =
                MemSlice::from_slice(unsafe { std::mem::transmute::<&[u8], &'static [u8]>(slice) });
            let ptr_out = mem_slice.as_ptr();

            assert_eq!(ptr_out, ptr);
        }
    }

    #[test]
    fn test_mem_slice_slicing() {
        use super::MemSlice;

        {
            let vec = vec![1u8, 2, 3, 4, 5];
            let slice = vec.as_slice();

            let mem_slice =
                MemSlice::from_slice(unsafe { std::mem::transmute::<&[u8], &'static [u8]>(slice) });

            let out = &*mem_slice.slice(3..5);
            assert_eq!(out, &slice[3..5]);
            assert_eq!(out.as_ptr(), slice[3..5].as_ptr());
        }
    }
}
