use crate::prelude::*;

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
}
