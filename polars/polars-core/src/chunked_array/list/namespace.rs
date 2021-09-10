use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;
use polars_arrow::prelude::ValueSize;

impl ListChunked {
    pub fn lst_max(&self) -> Series {
        self.apply_amortized(|s| s.as_ref().max_as_series())
            .explode()
            .unwrap()
            .into_series()
    }

    pub fn lst_min(&self) -> Series {
        self.apply_amortized(|s| s.as_ref().min_as_series())
            .explode()
            .unwrap()
            .into_series()
    }

    pub fn lst_sum(&self) -> Series {
        self.apply_amortized(|s| s.as_ref().sum_as_series())
            .explode()
            .unwrap()
            .into_series()
    }

    pub fn lst_mean(&self) -> Float64Chunked {
        self.amortized_iter()
            .map(|s| s.map(|s| s.as_ref().mean()).flatten())
            .collect()
    }

    pub fn lst_sort(&self, reverse: bool) -> ListChunked {
        self.apply_amortized(|s| s.as_ref().sort(reverse))
    }

    pub fn lst_reverse(&self) -> ListChunked {
        self.apply_amortized(|s| s.as_ref().reverse())
    }

    pub fn lst_unique(&self) -> Result<ListChunked> {
        self.try_apply_amortized(|s| s.as_ref().unique())
    }

    pub fn lst_lengths(&self) -> UInt32Chunked {
        let mut lengths = AlignedVec::with_capacity(self.len());
        self.downcast_iter().for_each(|arr| {
            let offsets = arr.offsets().as_slice();
            let mut last = offsets[0];
            for o in &offsets[1..] {
                lengths.push((*o - last) as u32);
                last = *o;
            }
        });
        UInt32Chunked::new_from_aligned_vec(self.name(), lengths)
    }

    pub fn lst_concat(&self, other: &[Series]) -> Result<ListChunked> {
        let mut iters = Vec::with_capacity(other.len() + 1);
        let dtype = self.dtype();
        let length = self.len();
        for s in other {
            if s.dtype() != dtype {
                return Err(PolarsError::DataTypeMisMatch(
                    format!("cannot concat {:?} into a list of {:?}", s.dtype(), dtype).into(),
                ));
            }
            if s.len() != length {
                return Err(PolarsError::ShapeMisMatch(
                    format!("length {} does not match {}", s.len(), length).into(),
                ));
            }
            iters.push(s.list()?.amortized_iter())
        }
        let mut first_iter = self.into_iter();
        let mut builder = get_list_builder(
            &self.inner_dtype(),
            self.get_values_size() * (other.len() + 1),
            self.len(),
            self.name(),
        );

        for _ in 0..self.len() {
            let mut acc = match first_iter.next().unwrap() {
                Some(s) => s,
                None => {
                    builder.append_null();
                    continue;
                }
            };
            for it in &mut iters {
                match it.next().unwrap() {
                    Some(s) => {
                        acc.append(s.as_ref())?;
                    }
                    None => {
                        builder.append_null();
                        continue;
                    }
                }
            }
            builder.append_series(&acc);
        }
        Ok(builder.finish())
    }
}
