use polars_utils::aliases::PlRandomState;
use polars_utils::hashing::BytesHash;
use rayon::prelude::*;

use crate::hashing::get_null_hash_value;
use crate::prelude::*;
use crate::utils::{_set_partition_size, _split_offsets};
use crate::POOL;

#[inline]
fn fill_bytes_hashes<'a, T>(
    ca: &'a ChunkedArray<T>,
    null_h: u64,
    hb: PlRandomState,
) -> Vec<BytesHash>
where
    T: PolarsDataType,
    <<T as PolarsDataType>::Array as StaticArray>::ValueT<'a>: AsRef<[u8]>,
{
    let mut byte_hashes = Vec::with_capacity(ca.len());
    for arr in ca.downcast_iter() {
        for opt_b in arr.iter() {
            let opt_b = opt_b.as_ref().map(|v| v.as_ref());
            // SAFETY:
            // the underlying data is tied to self
            let opt_b = unsafe { std::mem::transmute::<Option<&[u8]>, Option<&'a [u8]>>(opt_b) };
            let hash = match opt_b {
                Some(s) => hb.hash_one(s),
                None => null_h,
            };
            byte_hashes.push(BytesHash::new(opt_b, hash))
        }
    }
    byte_hashes
}

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
    for<'a> <T::Array as StaticArray>::ValueT<'a>: AsRef<[u8]>,
{
    #[allow(clippy::needless_lifetimes)]
    pub fn to_bytes_hashes<'a>(
        &'a self,
        mut multithreaded: bool,
        hb: PlRandomState,
    ) -> Vec<Vec<BytesHash<'a>>> {
        multithreaded &= POOL.current_num_threads() > 1;
        let null_h = get_null_hash_value(&hb);

        if multithreaded {
            let n_partitions = _set_partition_size();

            let split = _split_offsets(self.len(), n_partitions);

            POOL.install(|| {
                split
                    .into_par_iter()
                    .map(|(offset, len)| {
                        let ca = self.slice(offset as i64, len);
                        let byte_hashes = fill_bytes_hashes(&ca, null_h, hb.clone());

                        // SAFETY:
                        // the underlying data is tied to self
                        unsafe {
                            std::mem::transmute::<Vec<BytesHash<'_>>, Vec<BytesHash<'a>>>(
                                byte_hashes,
                            )
                        }
                    })
                    .collect::<Vec<_>>()
            })
        } else {
            vec![fill_bytes_hashes(self, null_h, hb.clone())]
        }
    }
}
