use crate::parquet::schema::types::PhysicalType;

/// A physical native representation of a Parquet fixed-sized type.
pub trait NativeType: std::fmt::Debug + Send + Sync + 'static + Copy + Clone {
    type Bytes: AsRef<[u8]>
        + bytemuck::Pod
        + IntoIterator<Item = u8>
        + for<'a> TryFrom<&'a [u8], Error = std::array::TryFromSliceError>
        + std::fmt::Debug
        + Clone
        + Copy;

    fn to_le_bytes(&self) -> Self::Bytes;

    fn from_le_bytes(bytes: Self::Bytes) -> Self;

    fn ord(&self, other: &Self) -> std::cmp::Ordering;

    const TYPE: PhysicalType;
}

macro_rules! native {
    ($type:ty, $physical_type:expr) => {
        impl NativeType for $type {
            type Bytes = [u8; std::mem::size_of::<Self>()];
            #[inline]
            fn to_le_bytes(&self) -> Self::Bytes {
                Self::to_le_bytes(*self)
            }

            #[inline]
            fn from_le_bytes(bytes: Self::Bytes) -> Self {
                Self::from_le_bytes(bytes)
            }

            #[inline]
            fn ord(&self, other: &Self) -> std::cmp::Ordering {
                self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
            }

            const TYPE: PhysicalType = $physical_type;
        }
    };
}

native!(i32, PhysicalType::Int32);
native!(i64, PhysicalType::Int64);
native!(f32, PhysicalType::Float);
native!(f64, PhysicalType::Double);

impl NativeType for [u32; 3] {
    const TYPE: PhysicalType = PhysicalType::Int96;

    type Bytes = [u8; std::mem::size_of::<Self>()];
    #[inline]
    fn to_le_bytes(&self) -> Self::Bytes {
        let mut bytes = [0; 12];
        let first = self[0].to_le_bytes();
        bytes[0] = first[0];
        bytes[1] = first[1];
        bytes[2] = first[2];
        bytes[3] = first[3];
        let second = self[1].to_le_bytes();
        bytes[4] = second[0];
        bytes[5] = second[1];
        bytes[6] = second[2];
        bytes[7] = second[3];
        let third = self[2].to_le_bytes();
        bytes[8] = third[0];
        bytes[9] = third[1];
        bytes[10] = third[2];
        bytes[11] = third[3];
        bytes
    }

    #[inline]
    fn from_le_bytes(bytes: Self::Bytes) -> Self {
        let mut first = [0; 4];
        first[0] = bytes[0];
        first[1] = bytes[1];
        first[2] = bytes[2];
        first[3] = bytes[3];
        let mut second = [0; 4];
        second[0] = bytes[4];
        second[1] = bytes[5];
        second[2] = bytes[6];
        second[3] = bytes[7];
        let mut third = [0; 4];
        third[0] = bytes[8];
        third[1] = bytes[9];
        third[2] = bytes[10];
        third[3] = bytes[11];
        [
            u32::from_le_bytes(first),
            u32::from_le_bytes(second),
            u32::from_le_bytes(third),
        ]
    }

    #[inline]
    fn ord(&self, other: &Self) -> std::cmp::Ordering {
        int96_to_i64_ns(*self).ord(&int96_to_i64_ns(*other))
    }
}

#[inline]
pub fn int96_to_i64_ns(value: [u32; 3]) -> i64 {
    const JULIAN_DAY_OF_EPOCH: i64 = 2_440_588;
    const SECONDS_PER_DAY: i64 = 86_400;
    const NANOS_PER_SECOND: i64 = 1_000_000_000;

    let day = value[2] as i64;
    let nanoseconds = ((value[1] as i64) << 32) + value[0] as i64;
    let seconds = (day - JULIAN_DAY_OF_EPOCH) * SECONDS_PER_DAY;

    seconds * NANOS_PER_SECOND + nanoseconds
}

/// Returns the ordering of two binary values.
pub fn ord_binary<'a>(a: &'a [u8], b: &'a [u8]) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    match (a.is_empty(), b.is_empty()) {
        (true, true) => return Equal,
        (true, false) => return Less,
        (false, true) => return Greater,
        (false, false) => {},
    }

    for (v1, v2) in a.iter().zip(b.iter()) {
        match v1.cmp(v2) {
            Equal => continue,
            other => return other,
        }
    }
    Equal
}

#[inline]
pub fn decode<T: NativeType>(chunk: &[u8]) -> T {
    assert!(chunk.len() >= std::mem::size_of::<<T as NativeType>::Bytes>());
    unsafe { decode_unchecked(chunk) }
}

/// Convert a Little-Endian byte-slice into the `T`
///
/// # Safety
///
/// This is safe if the length is properly checked.
#[inline]
pub unsafe fn decode_unchecked<T: NativeType>(chunk: &[u8]) -> T {
    let chunk: <T as NativeType>::Bytes = unsafe { chunk.try_into().unwrap_unchecked() };
    T::from_le_bytes(chunk)
}
