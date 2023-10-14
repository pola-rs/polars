use crate::bitmap::utils::{BitChunkIterExact, BitChunksExact};
use crate::bitmap::{chunk_iter_to_vec, Bitmap};

/// Apply a bitwise operation `op` to one input and return the result as a [`Bitmap`].
pub fn unary_mut<F>(lhs: &Bitmap, op: F) -> Bitmap
where
    F: FnMut(u64) -> u64,
{
    let (slice, offset, length) = lhs.as_slice();
    if offset == 0 {
        let iter = BitChunksExact::<u64>::new(slice, length);
        unary_impl(iter, op, lhs.len())
    } else {
        let iter = lhs.chunks::<u64>();
        unary_impl(iter, op, lhs.len())
    }
}

fn unary_impl<F, I>(iter: I, mut op: F, length: usize) -> Bitmap
where
    I: BitChunkIterExact<u64>,
    F: FnMut(u64) -> u64,
{
    let rem = op(iter.remainder());

    // TODO! this can be done without chaining
    let iterator = iter.map(op).chain(std::iter::once(rem));

    let buffer = chunk_iter_to_vec(iterator);

    Bitmap::from_u8_vec(buffer, length)
}
