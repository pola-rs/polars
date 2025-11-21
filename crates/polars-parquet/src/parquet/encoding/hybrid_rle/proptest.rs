use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Chunk {
    Rle(u32, usize),
    Bitpacked(Vec<u32>),
}

proptest::prop_compose! {
    fn hybrid_rle_chunks
        (max_idx: u32, size: usize)
        (idxs in proptest::collection::vec(0..=max_idx, size),
            mut chunk_offsets in proptest::collection::vec((0..=size, any::<bool>()), 2..=size.max(2)),
        )
    -> Vec<Chunk> {
        if size == 0 {
            return Vec::new();
        }

        chunk_offsets.sort_unstable();
        chunk_offsets.first_mut().unwrap().0 = 0;
        chunk_offsets.last_mut().unwrap().0 = idxs.len();
        chunk_offsets.dedup_by_key(|(idx, _)| *idx);

        chunk_offsets
            .windows(2)
            .map(|values| {
                let (start, is_bitpacked) = values[0];
                let (end, _) = values[1];

                if is_bitpacked {
                    Chunk::Bitpacked(idxs[start..end].to_vec())
                } else {
                    Chunk::Rle(idxs[start], end - start)
                }
            })
            .collect::<Vec<Chunk>>()
    }
}

proptest::prop_compose! {
    pub fn hybrid_rle
        (max_idx: u32, size: usize)
        (chunks in hybrid_rle_chunks(max_idx, size))
    -> Vec<u8> {
        use super::encoder::Encoder;
        let mut buffer = Vec::new();
        let bit_width = 32 - max_idx.leading_zeros();
        for chunk in chunks {
            match chunk {
                Chunk::Rle(value, size) => {
                    u32::run_length_encode(&mut buffer, size, value, bit_width).unwrap()
                }
                Chunk::Bitpacked(values) => {
                    u32::bitpacked_encode(&mut buffer, values.into_iter(), bit_width as usize).unwrap()
                }
            }
        }
        buffer
    }
}
