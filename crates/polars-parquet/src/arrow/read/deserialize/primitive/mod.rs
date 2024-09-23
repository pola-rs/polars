use arrow::types::NativeType;
use num_traits::AsPrimitive;

use crate::parquet::types::{decode, NativeType as ParquetNativeType};

mod float;
mod integer;

pub(crate) use float::FloatDecoder;
pub(crate) use integer::IntDecoder;

use super::utils::array_chunks::ArrayChunks;
use super::utils::BatchableCollector;
use super::ParquetResult;
use crate::parquet::encoding::delta_bitpacked::{self, DeltaGatherer};

#[derive(Debug)]
pub(crate) struct PrimitiveDecoder<P, T, D>
where
    P: ParquetNativeType,
    T: NativeType,
    D: DecoderFunction<P, T>,
{
    pub(crate) decoder: D,
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
    fn decode(self, x: P) -> T;
}

#[derive(Default, Clone, Copy)]
pub(crate) struct UnitDecoderFunction<T>(std::marker::PhantomData<T>);
impl<T: NativeType + ParquetNativeType> DecoderFunction<T, T> for UnitDecoderFunction<T> {
    #[inline(always)]
    fn decode(self, x: T) -> T {
        x
    }
}

#[derive(Default, Clone, Copy)]
pub(crate) struct AsDecoderFunction<P, T>(std::marker::PhantomData<(P, T)>);
macro_rules! as_decoder_impl {
    ($($p:ty => $t:ty,)+) => {
        $(
        impl DecoderFunction<$p, $t> for AsDecoderFunction<$p, $t> {
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
    #[inline(always)]
    fn decode(self, x: P) -> T {
        (self.0)(x)
    }
}

pub(crate) struct PlainDecoderFnCollector<'a, 'b, P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    pub(crate) chunks: &'b mut ArrayChunks<'a, P>,
    pub(crate) decoder: D,
    pub(crate) _pd: std::marker::PhantomData<T>,
}

impl<'a, 'b, P, T, D: DecoderFunction<P, T>> BatchableCollector<(), Vec<T>>
    for PlainDecoderFnCollector<'a, 'b, P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    fn reserve(target: &mut Vec<T>, n: usize) {
        target.reserve(n);
    }

    fn push_n(&mut self, target: &mut Vec<T>, n: usize) -> ParquetResult<()> {
        let n = usize::min(self.chunks.len(), n);
        let (items, remainder) = self.chunks.bytes.split_at(n);
        let decoder = self.decoder;
        target.extend(
            items
                .iter()
                .map(|chunk| decoder.decode(P::from_le_bytes(*chunk))),
        );
        self.chunks.bytes = remainder;
        Ok(())
    }

    fn push_n_nulls(&mut self, target: &mut Vec<T>, n: usize) -> ParquetResult<()> {
        target.resize(target.len() + n, T::default());
        Ok(())
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        self.chunks.skip_in_place(n);
        Ok(())
    }
}

fn deserialize_plain<P, T, D>(values: &[u8], decoder: D) -> Vec<T>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    values
        .chunks_exact(std::mem::size_of::<P>())
        .map(decode)
        .map(|v| decoder.decode(v))
        .collect::<Vec<_>>()
}

struct DeltaTranslator<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    dfn: D,
    _pd: std::marker::PhantomData<(P, T)>,
}

struct DeltaCollector<'a, 'b, P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    decoder: &'b mut delta_bitpacked::Decoder<'a>,
    gatherer: DeltaTranslator<P, T, D>,
}

impl<P, T, D> DeltaGatherer for DeltaTranslator<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    type Target = Vec<T>;

    fn target_len(&self, target: &Self::Target) -> usize {
        target.len()
    }

    fn target_reserve(&self, target: &mut Self::Target, n: usize) {
        target.reserve(n);
    }

    fn gather_one(&mut self, target: &mut Self::Target, v: i64) -> ParquetResult<()> {
        target.push(self.dfn.decode(v.as_()));
        Ok(())
    }

    fn gather_constant(
        &mut self,
        target: &mut Self::Target,
        v: i64,
        delta: i64,
        num_repeats: usize,
    ) -> ParquetResult<()> {
        target.extend((0..num_repeats).map(|i| self.dfn.decode((v + (i as i64) * delta).as_())));
        Ok(())
    }

    fn gather_slice(&mut self, target: &mut Self::Target, slice: &[i64]) -> ParquetResult<()> {
        target.extend(slice.iter().copied().map(|v| self.dfn.decode(v.as_())));
        Ok(())
    }

    fn gather_chunk(&mut self, target: &mut Self::Target, chunk: &[i64; 64]) -> ParquetResult<()> {
        target.extend(chunk.iter().copied().map(|v| self.dfn.decode(v.as_())));
        Ok(())
    }
}

impl<'a, 'b, P, T, D> BatchableCollector<(), Vec<T>> for DeltaCollector<'a, 'b, P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    fn reserve(target: &mut Vec<T>, n: usize) {
        target.reserve(n);
    }

    fn push_n(&mut self, target: &mut Vec<T>, n: usize) -> ParquetResult<()> {
        let start_length = target.len();
        let start_num_elems = self.decoder.len();

        self.decoder.gather_n_into(target, n, &mut self.gatherer)?;

        let consumed_elements = usize::min(n, start_num_elems);

        debug_assert_eq!(self.decoder.len(), start_num_elems - consumed_elements);
        debug_assert_eq!(target.len(), start_length + consumed_elements);

        Ok(())
    }

    fn push_n_nulls(&mut self, target: &mut Vec<T>, n: usize) -> ParquetResult<()> {
        target.resize(target.len() + n, T::default());
        Ok(())
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        self.decoder.skip_in_place(n)
    }
}
