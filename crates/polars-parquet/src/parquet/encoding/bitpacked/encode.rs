use super::{Unpackable, Unpacked};

/// Encodes (packs) a slice of [`Unpackable`] into bitpacked bytes `packed`, using `num_bits` per value.
///
/// This function assumes that the maximum value in `unpacked` fits in `num_bits` bits
/// and saturates higher values.
///
/// Only the first `ceil8(unpacked.len() * num_bits)` of `packed` are populated.
pub fn encode<T: Unpackable>(unpacked: &[T], num_bits: usize, packed: &mut [u8]) {
    let chunks = unpacked.chunks_exact(T::Unpacked::LENGTH);

    let remainder = chunks.remainder();

    let packed_size = (T::Unpacked::LENGTH * num_bits + 7) / 8;
    if !remainder.is_empty() {
        let packed_chunks = packed.chunks_mut(packed_size);
        let mut last_chunk = T::Unpacked::zero();
        for i in 0..remainder.len() {
            last_chunk[i] = remainder[i]
        }

        chunks
            .chain(std::iter::once(last_chunk.as_ref()))
            .zip(packed_chunks)
            .for_each(|(unpacked, packed)| {
                T::pack(&unpacked.try_into().unwrap(), num_bits, packed);
            });
    } else {
        let packed_chunks = packed.chunks_exact_mut(packed_size);
        chunks.zip(packed_chunks).for_each(|(unpacked, packed)| {
            T::pack(&unpacked.try_into().unwrap(), num_bits, packed);
        });
    }
}

/// Encodes (packs) a potentially incomplete pack of [`Unpackable`] into bitpacked
/// bytes `packed`, using `num_bits` per value.
///
/// This function assumes that the maximum value in `unpacked` fits in `num_bits` bits
/// and saturates higher values.
///
/// Only the first `ceil8(unpacked.len() * num_bits)` of `packed` are populated.
#[inline]
pub fn encode_pack<T: Unpackable>(unpacked: &[T], num_bits: usize, packed: &mut [u8]) {
    if unpacked.len() < T::Unpacked::LENGTH {
        let mut complete_unpacked = T::Unpacked::zero();
        complete_unpacked.as_mut()[..unpacked.len()].copy_from_slice(unpacked);
        T::pack(&complete_unpacked, num_bits, packed)
    } else {
        T::pack(&unpacked.try_into().unwrap(), num_bits, packed)
    }
}
