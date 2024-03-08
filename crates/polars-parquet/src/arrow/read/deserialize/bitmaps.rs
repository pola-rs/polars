use arrow::bitmap::MutableBitmap;
use polars_error::PolarsResult;

use crate::parquet::encoding::hybrid_rle;

#[derive(Debug)]
pub(crate) struct BitMapRleDecoder<'a> {
    decoder: hybrid_rle::Decoder<'a>,
    num_values: usize,
}

impl<'a> BitMapRleDecoder<'a> {
    pub(crate) fn try_new(values: &'a [u8], num_values: usize) -> PolarsResult<Self> {
        let decoder = hybrid_rle::Decoder::new(values, 1);
        Ok(Self {
            decoder,
            num_values,
        })
    }

    pub(crate) fn materialize(
        &mut self,
        remaining: usize,
        state: &mut MutableBitmap,
    ) -> PolarsResult<()> {
        let mut take = std::cmp::min(remaining, self.num_values);
        let taken = take;
        state.reserve(take);

        use hybrid_rle::HybridEncoded;
        while take > 0 {
            if let Some(res) = self.decoder.next() {
                match res? {
                    HybridEncoded::Bitpacked(pack) => {
                        let n = pack.len() * 8;
                        let n = std::cmp::min(n, take);
                        state.extend_from_slice(pack, 0, n);
                        take -= n;
                    },
                    HybridEncoded::Rle(val, n) => {
                        let is_set = val[0] == 1;
                        let n = std::cmp::min(n, take);
                        state.extend_constant(n, is_set);
                        take -= n;
                    },
                }
            } else {
                break;
            }
        }
        self.num_values -= taken;
        Ok(())
    }

    pub(crate) fn len(&self) -> usize {
        self.num_values
    }
}
