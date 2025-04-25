use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};

use bytemuck::{Pod, Zeroable};
use polars_error::*;
use polars_utils::min_max::MinMax;
use polars_utils::nulls::IsNull;
use polars_utils::total_ord::{TotalEq, TotalOrd};

use crate::datatypes::PrimitiveType;
use crate::types::{Bytes16Alignment4, NativeType};

// We use this instead of u128 because we want alignment of <= 8 bytes.
/// A reference to a set of bytes.
///
/// If `length <= 12`, these bytes are inlined over the `prefix`, `buffer_idx` and `offset` fields.
/// If `length > 12`, these fields specify a slice of a buffer.
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct View {
    /// The length of the string/bytes.
    pub length: u32,
    /// First 4 bytes of string/bytes data.
    pub prefix: u32,
    /// The buffer index.
    pub buffer_idx: u32,
    /// The offset into the buffer.
    pub offset: u32,
}

impl fmt::Debug for View {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.length <= Self::MAX_INLINE_SIZE {
            fmt.debug_struct("View")
                .field("length", &self.length)
                .field("content", &unsafe {
                    std::slice::from_raw_parts(
                        (self as *const _ as *const u8).add(4),
                        self.length as usize,
                    )
                })
                .finish()
        } else {
            fmt.debug_struct("View")
                .field("length", &self.length)
                .field("prefix", &self.prefix.to_be_bytes())
                .field("buffer_idx", &self.buffer_idx)
                .field("offset", &self.offset)
                .finish()
        }
    }
}

impl View {
    pub const MAX_INLINE_SIZE: u32 = 12;

    #[inline(always)]
    pub fn is_inline(&self) -> bool {
        self.length <= Self::MAX_INLINE_SIZE
    }

    #[inline(always)]
    pub fn as_u128(self) -> u128 {
        unsafe { std::mem::transmute(self) }
    }

    /// Create a new inline view without verifying the length
    ///
    /// # Safety
    ///
    /// It needs to hold that `bytes.len() <= View::MAX_INLINE_SIZE`.
    #[inline]
    pub unsafe fn new_inline_unchecked(bytes: &[u8]) -> Self {
        debug_assert!(bytes.len() <= u32::MAX as usize);
        debug_assert!(bytes.len() as u32 <= Self::MAX_INLINE_SIZE);

        let mut view = Self {
            length: bytes.len() as u32,
            ..Default::default()
        };

        let view_ptr = &mut view as *mut _ as *mut u8;

        // SAFETY:
        // - bytes length <= 12,
        // - size_of::<View> == 16
        // - View is laid out as [length, prefix, buffer_idx, offset] (using repr(C))
        // - By grabbing the view_ptr and adding 4, we have provenance over prefix, buffer_idx and
        // offset. (i.e. the same could not be achieved with &mut self.prefix as *mut _ as *mut u8)
        unsafe {
            let inline_data_ptr = view_ptr.add(4);
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), inline_data_ptr, bytes.len());
        }
        view
    }

    /// Create a new inline view
    ///
    /// # Panics
    ///
    /// Panics if the `bytes.len() > View::MAX_INLINE_SIZE`.
    #[inline]
    pub fn new_inline(bytes: &[u8]) -> Self {
        assert!(bytes.len() as u32 <= Self::MAX_INLINE_SIZE);
        unsafe { Self::new_inline_unchecked(bytes) }
    }

    /// Create a new inline view
    ///
    /// # Safety
    ///
    /// It needs to hold that `bytes.len() > View::MAX_INLINE_SIZE`.
    #[inline]
    pub unsafe fn new_noninline_unchecked(bytes: &[u8], buffer_idx: u32, offset: u32) -> Self {
        debug_assert!(bytes.len() <= u32::MAX as usize);
        debug_assert!(bytes.len() as u32 > View::MAX_INLINE_SIZE);

        // SAFETY: The invariant of this function guarantees that this is safe.
        let prefix = unsafe { u32::from_le_bytes(bytes[0..4].try_into().unwrap_unchecked()) };
        Self {
            length: bytes.len() as u32,
            prefix,
            buffer_idx,
            offset,
        }
    }

    #[inline]
    pub fn new_from_bytes(bytes: &[u8], buffer_idx: u32, offset: u32) -> Self {
        debug_assert!(bytes.len() <= u32::MAX as usize);

        // SAFETY: We verify the invariant with the outer if statement
        unsafe {
            if bytes.len() as u32 <= Self::MAX_INLINE_SIZE {
                Self::new_inline_unchecked(bytes)
            } else {
                Self::new_noninline_unchecked(bytes, buffer_idx, offset)
            }
        }
    }

    /// Constructs a byteslice from this view.
    ///
    /// # Safety
    /// Assumes that this view is valid for the given buffers.
    #[inline]
    pub unsafe fn get_slice_unchecked<'a, B: AsRef<[u8]>>(&'a self, buffers: &'a [B]) -> &'a [u8] {
        unsafe {
            if self.length <= Self::MAX_INLINE_SIZE {
                self.get_inlined_slice_unchecked()
            } else {
                self.get_external_slice_unchecked(buffers)
            }
        }
    }

    /// Construct a byte slice from an inline view, if it is inline.
    #[inline]
    pub fn get_inlined_slice(&self) -> Option<&[u8]> {
        if self.length <= Self::MAX_INLINE_SIZE {
            unsafe { Some(self.get_inlined_slice_unchecked()) }
        } else {
            None
        }
    }

    /// Construct a byte slice from an inline view.
    ///
    /// # Safety
    /// Assumes that this view is inlinable.
    #[inline]
    pub unsafe fn get_inlined_slice_unchecked(&self) -> &[u8] {
        debug_assert!(self.length <= View::MAX_INLINE_SIZE);
        let ptr = self as *const View as *const u8;
        unsafe { std::slice::from_raw_parts(ptr.add(4), self.length as usize) }
    }

    /// Construct a byte slice from an external view.
    ///
    /// # Safety
    /// Assumes that this view is in the external buffers.
    #[inline]
    pub unsafe fn get_external_slice_unchecked<'a, B: AsRef<[u8]>>(
        &self,
        buffers: &'a [B],
    ) -> &'a [u8] {
        debug_assert!(self.length > View::MAX_INLINE_SIZE);
        let data = buffers.get_unchecked(self.buffer_idx as usize);
        let offset = self.offset as usize;
        data.as_ref()
            .get_unchecked(offset..offset + self.length as usize)
    }

    /// Extend a `Vec<View>` with inline views slices of `src` with `width`.
    ///
    /// This tries to use SIMD to optimize the copying and can be massively faster than doing a
    /// `views.extend(src.chunks_exact(width).map(View::new_inline))`.
    ///
    /// # Panics
    ///
    /// This function panics if `src.len()` is not divisible by `width`, `width >
    /// View::MAX_INLINE_SIZE` or `width == 0`.
    pub fn extend_with_inlinable_strided(views: &mut Vec<Self>, src: &[u8], width: u8) {
        macro_rules! dispatch {
            ($n:ident = $match:ident in [$($v:literal),+ $(,)?] => $block:block, otherwise = $otherwise:expr) => {
                match $match {
                    $(
                        $v => {
                            const $n: usize = $v;

                            $block
                        }
                    )+
                    _ => $otherwise,
                }
            }
        }

        let width = width as usize;

        assert!(width > 0);
        assert!(width <= View::MAX_INLINE_SIZE as usize);

        assert_eq!(src.len() % width, 0);

        let num_values = src.len() / width;

        views.reserve(num_values);

        #[allow(unused_mut)]
        let mut src = src;

        dispatch! {
            N = width in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] => {
                #[cfg(feature = "simd")]
                {
                    macro_rules! repeat_with {
                        ($i:ident = [$($v:literal),+ $(,)?] => $block:block) => {
                            $({
                                const $i: usize = $v;

                                $block
                            })+
                        }
                    }

                    use std::simd::*;

                    // SAFETY: This is always allowed, since views.len() is always in the Vec
                    // buffer.
                    let mut dst = unsafe { views.as_mut_ptr().add(views.len()).cast::<u8>() };

                    let length_mask = u8x16::from_array([N as u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

                    const BLOCKS_PER_LOAD: usize = 16 / N;
                    const BYTES_PER_LOOP: usize = N * BLOCKS_PER_LOAD;

                    let num_loops = (src.len() / BYTES_PER_LOOP).saturating_sub(1);

                    for _ in 0..num_loops {
                        // SAFETY: The num_loops calculates how many times we can do this.
                        let loaded = u8x16::from_array(unsafe {
                            src.get_unchecked(..16).try_into().unwrap()
                        });
                        src = unsafe { src.get_unchecked(BYTES_PER_LOOP..) };

                        // This way we can reuse the same load for multiple views.
                        repeat_with!(
                            I = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] => {
                                if I < BLOCKS_PER_LOAD {
                                    let zero = u8x16::default();
                                    const SWIZZLE: [usize; 16] = const {
                                        let mut swizzle = [16usize; 16];

                                        let mut i = 0;
                                        while i < N {
                                            let idx = i + I * N;
                                            if idx < 16 {
                                                swizzle[4+i] = idx;
                                            }
                                            i += 1;
                                        }

                                        swizzle
                                    };

                                    let scattered = simd_swizzle!(loaded, zero, SWIZZLE);
                                    let view_bytes = (scattered | length_mask).to_array();

                                    // SAFETY: dst has the capacity reserved and view_bytes is 16
                                    // bytes long.
                                    unsafe {
                                        core::ptr::copy_nonoverlapping(view_bytes.as_ptr(), dst, 16);
                                        dst = dst.add(16);
                                    }
                                }
                            }
                        );
                    }

                    unsafe {
                        views.set_len(views.len() + num_loops * BLOCKS_PER_LOAD);
                    }
                }

                views.extend(src.chunks_exact(N).map(|slice| unsafe {
                    View::new_inline_unchecked(slice)
                }));
            },
            otherwise = unreachable!()
        }
    }
}

impl IsNull for View {
    const HAS_NULLS: bool = false;
    type Inner = Self;

    fn is_null(&self) -> bool {
        false
    }

    fn unwrap_inner(self) -> Self::Inner {
        self
    }
}

impl Display for View {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

unsafe impl Zeroable for View {}

unsafe impl Pod for View {}

impl PartialEq for View {
    fn eq(&self, other: &Self) -> bool {
        self.as_u128() == other.as_u128()
    }
}

// These are 'implemented' because we want to implement NativeType
// for View, that should probably not be done.
impl TotalOrd for View {
    fn tot_cmp(&self, _other: &Self) -> Ordering {
        unimplemented!()
    }
}

impl TotalEq for View {
    fn tot_eq(&self, other: &Self) -> bool {
        self.eq(other)
    }
}

impl MinMax for View {
    fn nan_min_lt(&self, _other: &Self) -> bool {
        unimplemented!()
    }

    fn nan_max_lt(&self, _other: &Self) -> bool {
        unimplemented!()
    }
}

impl NativeType for View {
    const PRIMITIVE: PrimitiveType = PrimitiveType::UInt128;

    type Bytes = [u8; 16];
    type AlignedBytes = Bytes16Alignment4;

    #[inline]
    fn to_le_bytes(&self) -> Self::Bytes {
        self.as_u128().to_le_bytes()
    }

    #[inline]
    fn to_be_bytes(&self) -> Self::Bytes {
        self.as_u128().to_be_bytes()
    }

    #[inline]
    fn from_le_bytes(bytes: Self::Bytes) -> Self {
        Self::from(u128::from_le_bytes(bytes))
    }

    #[inline]
    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        Self::from(u128::from_be_bytes(bytes))
    }
}

impl From<u128> for View {
    #[inline]
    fn from(value: u128) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl From<View> for u128 {
    #[inline]
    fn from(value: View) -> Self {
        value.as_u128()
    }
}

pub fn validate_views<B: AsRef<[u8]>, F>(
    views: &[View],
    buffers: &[B],
    validate_bytes: F,
) -> PolarsResult<()>
where
    F: Fn(&[u8]) -> PolarsResult<()>,
{
    for view in views {
        if let Some(inline_slice) = view.get_inlined_slice() {
            if view.length < View::MAX_INLINE_SIZE && view.as_u128() >> (32 + view.length * 8) != 0
            {
                polars_bail!(ComputeError: "view contained non-zero padding in prefix");
            }

            validate_bytes(inline_slice)?;
        } else {
            let data = buffers.get(view.buffer_idx as usize).ok_or_else(|| {
                polars_err!(OutOfBounds: "view index out of bounds\n\nGot: {} buffers and index: {}", buffers.len(), view.buffer_idx)
            })?;

            let start = view.offset as usize;
            let end = start + view.length as usize;
            let b = data
                .as_ref()
                .get(start..end)
                .ok_or_else(|| polars_err!(OutOfBounds: "buffer slice out of bounds"))?;

            polars_ensure!(b.starts_with(&view.prefix.to_le_bytes()), ComputeError: "prefix does not match string data");
            validate_bytes(b)?;
        }
    }

    Ok(())
}

pub fn validate_binary_views<B: AsRef<[u8]>>(views: &[View], buffers: &[B]) -> PolarsResult<()> {
    validate_views(views, buffers, |_| Ok(()))
}

fn validate_utf8(b: &[u8]) -> PolarsResult<()> {
    match simdutf8::basic::from_utf8(b) {
        Ok(_) => Ok(()),
        Err(_) => Err(polars_err!(ComputeError: "invalid utf8")),
    }
}

pub fn validate_utf8_views<B: AsRef<[u8]>>(views: &[View], buffers: &[B]) -> PolarsResult<()> {
    validate_views(views, buffers, validate_utf8)
}

/// Checks the views for valid UTF-8. Assumes the first num_trusted_buffers are
/// valid UTF-8 without checking.
/// # Safety
/// The views and buffers must uphold the invariants of BinaryView otherwise we will go OOB.
pub unsafe fn validate_views_utf8_only<B: AsRef<[u8]>>(
    views: &[View],
    buffers: &[B],
    mut num_trusted_buffers: usize,
) -> PolarsResult<()> {
    unsafe {
        while num_trusted_buffers < buffers.len()
            && buffers[num_trusted_buffers].as_ref().is_ascii()
        {
            num_trusted_buffers += 1;
        }

        // Fast path if all buffers are ASCII (or there are no buffers).
        if num_trusted_buffers >= buffers.len() {
            for view in views {
                if let Some(inlined_slice) = view.get_inlined_slice() {
                    validate_utf8(inlined_slice)?;
                }
            }
        } else {
            for view in views {
                if view.length <= View::MAX_INLINE_SIZE
                    || view.buffer_idx as usize >= num_trusted_buffers
                {
                    validate_utf8(view.get_slice_unchecked(buffers))?;
                }
            }
        }

        Ok(())
    }
}
