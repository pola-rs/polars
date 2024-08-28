use std::any::Any;

use polars_error::constants::LENGTH_LIMIT_MSG;

use crate::prelude::compare_inner::{IntoTotalEqInner, TotalEqInner};
use crate::prelude::*;
use crate::series::private::{PrivateSeries, PrivateSeriesNumeric};
use crate::series::*;

impl Series {
    pub fn new_null(name: &str, len: usize) -> Series {
        NullChunked::new(Arc::from(name), len).into_series()
    }
}

#[derive(Clone)]
pub struct NullChunked {
    pub(crate) name: Arc<str>,
    length: IdxSize,
    // we still need chunks as many series consumers expect
    // chunks to be there
    chunks: Vec<ArrayRef>,
}

impl NullChunked {
    pub(crate) fn new(name: Arc<str>, len: usize) -> Self {
        Self {
            name,
            length: len as IdxSize,
            chunks: vec![Box::new(arrow::array::NullArray::new(
                ArrowDataType::Null,
                len,
            ))],
        }
    }
}
impl PrivateSeriesNumeric for NullChunked {
    fn bit_repr(&self) -> Option<BitRepr> {
        Some(BitRepr::Small(UInt32Chunked::full_null(
            self.name.as_ref(),
            self.len(),
        )))
    }
}

impl PrivateSeries for NullChunked {
    fn compute_len(&mut self) {
        fn inner(chunks: &[ArrayRef]) -> usize {
            match chunks.len() {
                // fast path
                1 => chunks[0].len(),
                _ => chunks.iter().fold(0, |acc, arr| acc + arr.len()),
            }
        }
        self.length = IdxSize::try_from(inner(&self.chunks)).expect(LENGTH_LIMIT_MSG);
    }
    fn _field(&self) -> Cow<Field> {
        Cow::Owned(Field::new(self.name(), DataType::Null))
    }

    #[allow(unused)]
    fn _set_flags(&mut self, flags: MetadataFlags) {}

    fn _dtype(&self) -> &DataType {
        &DataType::Null
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        let len = match (self.len(), mask.len(), other.len()) {
            (a, b, c) if a == b && b == c => a,
            (1, a, b) | (a, 1, b) | (a, b, 1) if a == b => a,
            (a, 1, 1) | (1, a, 1) | (1, 1, a) => a,
            (_, 0, _) => 0,
            _ => {
                polars_bail!(ShapeMismatch: "shapes of `self`, `mask` and `other` are not suitable for `zip_with` operation")
            },
        };

        Ok(Self::new(self.name().into(), len).into_series())
    }
    fn subtract(&self, _rhs: &Series) -> PolarsResult<Series> {
        null_arithmetic(self, _rhs, "subtract")
    }

    fn add_to(&self, _rhs: &Series) -> PolarsResult<Series> {
        null_arithmetic(self, _rhs, "add_to")
    }
    fn multiply(&self, _rhs: &Series) -> PolarsResult<Series> {
        null_arithmetic(self, _rhs, "multiply")
    }
    fn divide(&self, _rhs: &Series) -> PolarsResult<Series> {
        null_arithmetic(self, _rhs, "divide")
    }
    fn remainder(&self, _rhs: &Series) -> PolarsResult<Series> {
        null_arithmetic(self, _rhs, "remainder")
    }

    #[cfg(feature = "algorithm_group_by")]
    fn group_tuples(&self, _multithreaded: bool, _sorted: bool) -> PolarsResult<GroupsProxy> {
        Ok(if self.is_empty() {
            GroupsProxy::default()
        } else {
            GroupsProxy::Slice {
                groups: vec![[0, self.length]],
                rolling: false,
            }
        })
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        AggList::agg_list(self, groups)
    }

    fn _get_flags(&self) -> MetadataFlags {
        MetadataFlags::empty()
    }

    fn vec_hash(&self, random_state: PlRandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        VecHash::vec_hash(self, random_state, buf)?;
        Ok(())
    }

    fn vec_hash_combine(
        &self,
        build_hasher: PlRandomState,
        hashes: &mut [u64],
    ) -> PolarsResult<()> {
        VecHash::vec_hash_combine(self, build_hasher, hashes)?;
        Ok(())
    }

    fn into_total_eq_inner<'a>(&'a self) -> Box<dyn TotalEqInner + 'a> {
        IntoTotalEqInner::into_total_eq_inner(self)
    }
}

fn null_arithmetic(lhs: &NullChunked, rhs: &Series, op: &str) -> PolarsResult<Series> {
    let output_len = match (lhs.len(), rhs.len()) {
        (1, len_r) => len_r,
        (len_l, 1) => len_l,
        (len_l, len_r) if len_l == len_r => len_l,
        _ => polars_bail!(ComputeError: "Cannot {:?} two series of different lengths.", op),
    };
    Ok(NullChunked::new(lhs.name().into(), output_len).into_series())
}

impl SeriesTrait for NullChunked {
    fn name(&self) -> &str {
        self.name.as_ref()
    }

    fn rename(&mut self, name: &str) {
        self.name = Arc::from(name)
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        &self.chunks
    }
    unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        &mut self.chunks
    }

    fn chunk_lengths(&self) -> ChunkLenIter {
        self.chunks.iter().map(|chunk| chunk.len())
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        Ok(NullChunked::new(self.name.clone(), indices.len()).into_series())
    }

    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Series {
        NullChunked::new(self.name.clone(), indices.len()).into_series()
    }

    fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Series> {
        Ok(NullChunked::new(self.name.clone(), indices.len()).into_series())
    }

    unsafe fn take_slice_unchecked(&self, indices: &[IdxSize]) -> Series {
        NullChunked::new(self.name.clone(), indices.len()).into_series()
    }

    fn len(&self) -> usize {
        self.length as usize
    }

    fn has_nulls(&self) -> bool {
        self.len() > 0
    }

    fn rechunk(&self) -> Series {
        NullChunked::new(self.name.clone(), self.len()).into_series()
    }

    fn drop_nulls(&self) -> Series {
        NullChunked::new(self.name.clone(), 0).into_series()
    }

    fn cast(&self, data_type: &DataType, _cast_options: CastOptions) -> PolarsResult<Series> {
        Ok(Series::full_null(self.name.as_ref(), self.len(), data_type))
    }

    fn null_count(&self) -> usize {
        self.len()
    }

    #[cfg(feature = "algorithm_group_by")]
    fn unique(&self) -> PolarsResult<Series> {
        let ca = NullChunked::new(self.name.clone(), self.n_unique().unwrap());
        Ok(ca.into_series())
    }

    #[cfg(feature = "algorithm_group_by")]
    fn n_unique(&self) -> PolarsResult<usize> {
        let n = if self.is_empty() { 0 } else { 1 };
        Ok(n)
    }

    fn new_from_index(&self, _index: usize, length: usize) -> Series {
        NullChunked::new(self.name.clone(), length).into_series()
    }

    fn get(&self, index: usize) -> PolarsResult<AnyValue> {
        polars_ensure!(index < self.len(), oob = index, self.len());
        Ok(AnyValue::Null)
    }

    unsafe fn get_unchecked(&self, _index: usize) -> AnyValue {
        AnyValue::Null
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        let (chunks, len) = chunkops::slice(&self.chunks, offset, length, self.len());
        NullChunked {
            name: self.name.clone(),
            length: len as IdxSize,
            chunks,
        }
        .into_series()
    }

    fn split_at(&self, offset: i64) -> (Series, Series) {
        let (l, r) = chunkops::split_at(self.chunks(), offset, self.len());
        (
            NullChunked {
                name: self.name.clone(),
                length: l.iter().map(|arr| arr.len() as IdxSize).sum(),
                chunks: l,
            }
            .into_series(),
            NullChunked {
                name: self.name.clone(),
                length: r.iter().map(|arr| arr.len() as IdxSize).sum(),
                chunks: r,
            }
            .into_series(),
        )
    }

    fn sort_with(&self, _options: SortOptions) -> PolarsResult<Series> {
        Ok(self.clone().into_series())
    }

    fn is_null(&self) -> BooleanChunked {
        BooleanChunked::full(self.name(), true, self.len())
    }

    fn is_not_null(&self) -> BooleanChunked {
        BooleanChunked::full(self.name(), false, self.len())
    }

    fn reverse(&self) -> Series {
        self.clone().into_series()
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        let len = if self.is_empty() {
            // We still allow a length of `1` because it could be `lit(true)`.
            polars_ensure!(filter.len() <= 1, ShapeMismatch: "filter's length: {} differs from that of the series: 0", filter.len());
            0
        } else {
            polars_ensure!(filter.len() == self.len(), ShapeMismatch: "filter's length: {} differs from that of the series: {}", filter.len(), self.len());
            filter.sum().unwrap_or(0) as usize
        };
        Ok(NullChunked::new(self.name.clone(), len).into_series())
    }

    fn shift(&self, _periods: i64) -> Series {
        self.clone().into_series()
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(other.dtype() == &DataType::Null, ComputeError: "expected null dtype");
        // we don't create a new null array to keep probability of aligned chunks higher
        self.chunks.extend(other.chunks().iter().cloned());
        self.length += other.len() as IdxSize;
        Ok(())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        *self = NullChunked::new(self.name.clone(), self.len() + other.len());
        Ok(())
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

unsafe impl IntoSeries for NullChunked {
    fn into_series(self) -> Series
    where
        Self: Sized,
    {
        Series(Arc::new(self))
    }
}
