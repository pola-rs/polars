mod decode;
mod encode;
mod pack;
mod unpack;

pub use decode::Decoder;
pub use encode::{encode, encode_pack};

/// A byte slice (e.g. `[u8; 8]`) denoting types that represent complete packs.
pub trait Packed:
    Copy
    + Sized
    + AsRef<[u8]>
    + AsMut<[u8]>
    + std::ops::IndexMut<usize, Output = u8>
    + for<'a> TryFrom<&'a [u8]>
{
    const LENGTH: usize;
    fn zero() -> Self;
}

impl Packed for [u8; 8] {
    const LENGTH: usize = 8;
    #[inline]
    fn zero() -> Self {
        [0; 8]
    }
}

impl Packed for [u8; 16 * 2] {
    const LENGTH: usize = 16 * 2;
    #[inline]
    fn zero() -> Self {
        [0; 16 * 2]
    }
}

impl Packed for [u8; 32 * 4] {
    const LENGTH: usize = 32 * 4;
    #[inline]
    fn zero() -> Self {
        [0; 32 * 4]
    }
}

impl Packed for [u8; 64 * 64] {
    const LENGTH: usize = 64 * 64;
    #[inline]
    fn zero() -> Self {
        [0; 64 * 64]
    }
}

/// A byte slice of [`Unpackable`] denoting complete unpacked arrays.
pub trait Unpacked<T>:
    Copy
    + Sized
    + AsRef<[T]>
    + AsMut<[T]>
    + std::ops::Index<usize, Output = T>
    + std::ops::IndexMut<usize, Output = T>
    + for<'a> TryFrom<&'a [T], Error = std::array::TryFromSliceError>
{
    const LENGTH: usize;
    fn zero() -> Self;
}

impl Unpacked<u8> for [u8; 8] {
    const LENGTH: usize = 8;
    #[inline]
    fn zero() -> Self {
        [0; 8]
    }
}

impl Unpacked<u16> for [u16; 16] {
    const LENGTH: usize = 16;
    #[inline]
    fn zero() -> Self {
        [0; 16]
    }
}

impl Unpacked<u32> for [u32; 32] {
    const LENGTH: usize = 32;
    #[inline]
    fn zero() -> Self {
        [0; 32]
    }
}

impl Unpacked<u64> for [u64; 64] {
    const LENGTH: usize = 64;
    #[inline]
    fn zero() -> Self {
        [0; 64]
    }
}

/// A type representing a type that can be bitpacked and unpacked by this crate.
pub trait Unpackable: Copy + Sized + Default {
    type Packed: Packed;
    type Unpacked: Unpacked<Self>;
    fn unpack(packed: &[u8], num_bits: usize, unpacked: &mut Self::Unpacked);
    fn pack(unpacked: &Self::Unpacked, num_bits: usize, packed: &mut [u8]);
}

impl Unpackable for u8 {
    type Packed = [u8; 8];
    type Unpacked = [u8; 8];

    #[inline]
    fn unpack(packed: &[u8], num_bits: usize, unpacked: &mut Self::Unpacked) {
        unpack::unpack8(packed, unpacked, num_bits)
    }

    #[inline]
    fn pack(packed: &Self::Unpacked, num_bits: usize, unpacked: &mut [u8]) {
        pack::pack8(packed, unpacked, num_bits)
    }
}

impl Unpackable for u16 {
    type Packed = [u8; 16 * 2];
    type Unpacked = [u16; 16];

    #[inline]
    fn unpack(packed: &[u8], num_bits: usize, unpacked: &mut Self::Unpacked) {
        unpack::unpack16(packed, unpacked, num_bits)
    }

    #[inline]
    fn pack(packed: &Self::Unpacked, num_bits: usize, unpacked: &mut [u8]) {
        pack::pack16(packed, unpacked, num_bits)
    }
}

impl Unpackable for u32 {
    type Packed = [u8; 32 * 4];
    type Unpacked = [u32; 32];

    #[inline]
    fn unpack(packed: &[u8], num_bits: usize, unpacked: &mut Self::Unpacked) {
        unpack::unpack32(packed, unpacked, num_bits)
    }

    #[inline]
    fn pack(packed: &Self::Unpacked, num_bits: usize, unpacked: &mut [u8]) {
        pack::pack32(packed, unpacked, num_bits)
    }
}

impl Unpackable for u64 {
    type Packed = [u8; 64 * 64];
    type Unpacked = [u64; 64];

    #[inline]
    fn unpack(packed: &[u8], num_bits: usize, unpacked: &mut Self::Unpacked) {
        unpack::unpack64(packed, unpacked, num_bits)
    }

    #[inline]
    fn pack(packed: &Self::Unpacked, num_bits: usize, unpacked: &mut [u8]) {
        pack::pack64(packed, unpacked, num_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn case1() -> (usize, Vec<u32>, Vec<u8>) {
        let num_bits = 3;
        let compressed = vec![
            0b10001000u8,
            0b11000110,
            0b11111010,
            0b10001000u8,
            0b11000110,
            0b11111010,
            0b10001000u8,
            0b11000110,
            0b11111010,
            0b10001000u8,
            0b11000110,
            0b11111010,
            0b10001000u8,
            0b11000110,
            0b11111010,
        ];
        let decompressed = vec![
            0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4,
            5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
        ];
        (num_bits, decompressed, compressed)
    }

    #[test]
    fn encode_large() {
        let (num_bits, unpacked, expected) = case1();
        let mut packed = vec![0u8; 4 * 32];

        encode(&unpacked, num_bits, &mut packed);
        assert_eq!(&packed[..15], expected);
    }

    #[test]
    fn test_encode() {
        let num_bits = 3;
        let unpacked = vec![0, 1, 2, 3, 4, 5, 6, 7];

        let mut packed = vec![0u8; 4 * 32];

        encode::<u32>(&unpacked, num_bits, &mut packed);

        let expected = vec![0b10001000u8, 0b11000110, 0b11111010];

        assert_eq!(&packed[..3], expected);
    }
}
