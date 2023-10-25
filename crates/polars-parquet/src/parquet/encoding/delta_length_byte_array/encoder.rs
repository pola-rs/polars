use crate::encoding::delta_bitpacked;

/// Encodes a clonable iterator of `&[u8]` into `buffer`. This does not allocated on the heap.
/// # Implementation
/// This encoding is equivalent to call [`delta_bitpacked::encode`] on the lengths of the items
/// of the iterator followed by extending the buffer from each item of the iterator.
pub fn encode<A: AsRef<[u8]>, I: Iterator<Item = A> + Clone>(iterator: I, buffer: &mut Vec<u8>) {
    let mut total_length = 0;
    delta_bitpacked::encode(
        iterator.clone().map(|x| {
            let len = x.as_ref().len();
            total_length += len;
            len as i64
        }),
        buffer,
    );
    buffer.reserve(total_length);
    iterator.for_each(|x| buffer.extend(x.as_ref()))
}
