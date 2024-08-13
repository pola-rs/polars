use arrow::types::NativeType;
use num_traits::AsPrimitive;

use crate::parquet::types::NativeType as ParquetNativeType;

mod basic;
mod integer;

pub(crate) use basic::PrimitiveDecoder;
pub(crate) use integer::IntDecoder;

use self::basic::DecoderFunction;
use super::utils::BatchableCollector;
use super::ParquetResult;
use crate::parquet::encoding::delta_bitpacked::{self, DeltaGatherer};

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
}
