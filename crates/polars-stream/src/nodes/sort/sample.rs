use polars_core::chunked_array::ops::sort::options::SortOptions;
use polars_core::datatypes::{BinaryChunked, DataType};
use polars_core::series::{IntoSeries, Series};
use polars_error::PolarsResult;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;
use rand::{Rng, RngExt};

/// A strided sample of the physical key column of one input pipe.
///
/// One row is kept per stride window, at a random offset within that window so
/// that a periodic key pattern does not bias the sample. Once the sample grows
/// past twice its target size it is halved and the stride doubled, keeping
/// between `sample_rows` and `2 * sample_rows` rows spread over everything
/// seen so far.
pub struct KeySample {
    stride: usize,
    /// Rows consumed so far.
    seen: usize,
    /// Absolute index of the next row to take.
    next: usize,
    parts: Vec<Series>,
    len: usize,
    target: usize,
    idxs: Vec<IdxSize>,
}

impl KeySample {
    pub fn new(sample_rows: usize) -> Self {
        Self {
            stride: 1,
            seen: 0,
            next: 0,
            parts: Vec::new(),
            len: 0,
            target: sample_rows.max(1),
            idxs: Vec::new(),
        }
    }

    /// Samples the physical key column of one morsel.
    pub fn add(&mut self, keys: &Series) {
        let height = keys.len();
        if height == 0 {
            return;
        }

        let mut rng = rand::rng();
        self.idxs.clear();
        while self.next < self.seen + height {
            self.idxs.push((self.next - self.seen) as IdxSize);
            let window_end = (self.next / self.stride + 1) * self.stride;
            self.next = window_end + rng.random_range(0..self.stride);
        }
        self.seen += height;

        if !self.idxs.is_empty() {
            // SAFETY: the indices are row offsets within this morsel.
            let part = unsafe { keys.take_slice_unchecked(&self.idxs) };
            self.parts.push(deshare(part));
            self.len += self.idxs.len();
        }

        while self.len >= 2 * self.target {
            self.halve(&mut rng);
        }
    }

    /// Returns the sampled rows, thinned out so that every row stands for
    /// `stride` input rows. Samples of pipes that saw different amounts of
    /// input then combine into a uniform sample of the whole input.
    ///
    /// `stride` must be at least the stride of this sample.
    pub fn into_parts_with_stride(mut self, stride: usize) -> Vec<Series> {
        let mut rng = rand::rng();
        while self.stride < stride && self.len > 0 {
            self.halve(&mut rng);
        }
        self.parts
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Keeps one row of every consecutive pair, at a random position.
    fn halve(&mut self, rng: &mut impl Rng) {
        let s = concat(core::mem::take(&mut self.parts));
        let idxs: Vec<IdxSize> = (0..self.len / 2)
            .map(|i| (2 * i + rng.random_range(0..2)) as IdxSize)
            .collect();
        // SAFETY: the indices are below the length of the concatenated sample.
        self.parts = vec![unsafe { s.take_slice_unchecked(&idxs) }];
        self.len = idxs.len();
        self.stride *= 2;
    }
}

/// Copies the bytes of a `Binary` sample so that it does not keep the buffers
/// of its morsel alive.
fn deshare(part: Series) -> Series {
    match part.dtype() {
        DataType::Binary => BinaryChunked::from_chunk_iter(
            part.name().clone(),
            part.binary()
                .unwrap()
                .downcast_iter()
                .map(|arr| arr.deshare()),
        )
        .into_series(),
        _ => part,
    }
}

fn concat(parts: Vec<Series>) -> Series {
    let mut parts = parts.into_iter();
    let mut out = parts.next().unwrap();
    for part in parts {
        out.append_owned(part).unwrap();
    }
    out
}

/// Computes the `b - 1` split keys separating `b` equally sized buckets.
///
/// The result is sorted in the order the buckets are laid out, so descending
/// when `descending` is set, and contains no nulls. Equal neighbours are kept:
/// they are what marks a key value that holds a whole bucket by itself.
pub fn split_keys(
    samples: Vec<Series>,
    dtype: &DataType,
    descending: bool,
    b: usize,
) -> PolarsResult<Series> {
    let samples: Vec<Series> = samples.into_iter().filter(|s| !s.is_empty()).collect();
    if samples.is_empty() || b <= 1 {
        return Ok(Series::new_empty(PlSmallStr::EMPTY, dtype));
    }

    let s = concat(samples).drop_nulls().sort_with(SortOptions {
        descending,
        nulls_last: false,
        multithreaded: true,
        maintain_order: false,
        limit: None,
    })?;

    let n = s.len();
    if n == 0 {
        return Ok(Series::new_empty(PlSmallStr::EMPTY, dtype));
    }

    let idxs: Vec<IdxSize> = (1..b).map(|j| (j * n / b) as IdxSize).collect();
    s.take_slice(&idxs)
}
