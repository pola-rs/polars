use memchr::memmem;
use polars_core::prelude::arity::{broadcast_binary_elementwise, unary_elementwise};
use polars_core::prelude::*;

/// The offset of the first occurrence of each `pat` in `ca`, or null if there is none.
///
/// `to_offset` converts the byte offset of a match in a value to the returned offset.
pub(crate) fn find_literal_offsets(
    ca: &BinaryChunked,
    pat: &BinaryChunked,
    to_offset: impl Fn(&[u8], usize) -> usize,
) -> UInt32Chunked {
    if pat.len() == 1 {
        let Some(pat) = pat.get(0) else {
            return UInt32Chunked::full_null(ca.name().clone(), ca.len());
        };
        let finder = memmem::Finder::new(pat);
        unary_elementwise(ca, |s: Option<&[u8]>| {
            let s = s?;
            finder.find(s).map(|i| to_offset(s, i) as u32)
        })
    } else {
        broadcast_binary_elementwise(ca, pat, |s: Option<&[u8]>, pat: Option<&[u8]>| {
            let (s, pat) = (s?, pat?);
            memmem::find(s, pat).map(|i| to_offset(s, i) as u32)
        })
    }
}
