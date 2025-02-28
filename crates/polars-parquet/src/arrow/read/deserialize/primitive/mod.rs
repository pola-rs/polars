use arrow::types::NativeType;

use crate::parquet::types::NativeType as ParquetNativeType;

mod float;
mod integer;
pub(crate) mod plain;

pub(crate) use float::FloatDecoder;
pub(crate) use integer::IntDecoder;

#[derive(Debug)]
pub(crate) struct PrimitiveDecoder<P, T, D>
where
    P: ParquetNativeType,
    T: NativeType,
    D: DecoderFunction<P, T>,
{
    pub(crate) decoder: D,
    pub(crate) intermediate: Vec<P>,
    _pd: std::marker::PhantomData<(P, T)>,
}

impl<P, T, D> PrimitiveDecoder<P, T, D>
where
    P: ParquetNativeType,
    T: NativeType,
    D: DecoderFunction<P, T>,
{
    #[inline]
    pub(crate) fn new(decoder: D) -> Self {
        Self {
            decoder,
            intermediate: Vec::new(),
            _pd: std::marker::PhantomData,
        }
    }
}

/// A function that defines how to decode from the
/// [`parquet::types::NativeType`][ParquetNativeType] to the [`arrow::types::NativeType`].
///
/// This should almost always be inlined.
pub(crate) trait DecoderFunction<P, T>: Copy
where
    T: NativeType,
    P: ParquetNativeType,
{
    const NEED_TO_DECODE: bool;
    const CAN_TRANSMUTE: bool = {
        let has_same_size = size_of::<P>() == size_of::<T>();
        let has_same_alignment = align_of::<P>() == align_of::<T>();

        has_same_size && has_same_alignment
    };

    fn decode(self, x: P) -> T;
}

#[derive(Default, Clone, Copy)]
pub(crate) struct UnitDecoderFunction<T>(std::marker::PhantomData<T>);
impl<T: NativeType + ParquetNativeType> DecoderFunction<T, T> for UnitDecoderFunction<T> {
    const NEED_TO_DECODE: bool = false;

    #[inline(always)]
    fn decode(self, x: T) -> T {
        x
    }
}

#[derive(Default, Clone, Copy)]
pub(crate) struct AsDecoderFunction<P: ParquetNativeType, T: NativeType>(
    std::marker::PhantomData<(P, T)>,
);
macro_rules! as_decoder_impl {
    ($($p:ty => $t:ty,)+) => {
        $(
        impl DecoderFunction<$p, $t> for AsDecoderFunction<$p, $t> {
            const NEED_TO_DECODE: bool = Self::CAN_TRANSMUTE;

            #[inline(always)]
            fn decode(self, x : $p) -> $t {
                x as $t
            }
        }
        )+
    };
}

as_decoder_impl![
    i32 => i8,
    i32 => i16,
    i32 => u8,
    i32 => u16,
    i32 => u32,
    i64 => i32,
    i64 => u32,
    i64 => u64,
];

#[derive(Default, Clone, Copy)]
pub(crate) struct IntoDecoderFunction<P, T>(std::marker::PhantomData<(P, T)>);
impl<P, T> DecoderFunction<P, T> for IntoDecoderFunction<P, T>
where
    P: ParquetNativeType + Into<T>,
    T: NativeType,
{
    const NEED_TO_DECODE: bool = true;

    #[inline(always)]
    fn decode(self, x: P) -> T {
        x.into()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ClosureDecoderFunction<P, T, F>(F, std::marker::PhantomData<(P, T)>);
impl<P, T, F> DecoderFunction<P, T> for ClosureDecoderFunction<P, T, F>
where
    P: ParquetNativeType,
    T: NativeType,
    F: Copy + Fn(P) -> T,
{
    const NEED_TO_DECODE: bool = true;

    #[inline(always)]
    fn decode(self, x: P) -> T {
        (self.0)(x)
    }
}
