use super::*;
use crate::chunked_array::comparison::*;
use crate::prelude::*;

unsafe impl<T: PolarsCategoricalType> IntoSeries for CategoricalChunked<T> {
    fn into_series(self) -> Series {
        // We do this hack to go from generic T to concrete T to avoid adding bounds on IntoSeries.
        with_match_categorical_physical_type!(T::physical(), |$C| {
            unsafe {
                Series(Arc::new(SeriesWrap(core::mem::transmute::<Self, CategoricalChunked<$C>>(self))))
            }
        })
    }
}

impl<T: PolarsCategoricalType> SeriesWrap<CategoricalChunked<T>> {
    unsafe fn apply_on_phys<F>(&self, apply: F) -> CategoricalChunked<T>
    where
        F: Fn(&ChunkedArray<T::PolarsPhysical>) -> ChunkedArray<T::PolarsPhysical>,
    {
        let cats = apply(self.0.physical());
        unsafe { CategoricalChunked::from_cats_and_dtype_unchecked(cats, self.0.dtype().clone()) }
    }

    unsafe fn try_apply_on_phys<F>(&self, apply: F) -> PolarsResult<CategoricalChunked<T>>
    where
        F: Fn(&ChunkedArray<T::PolarsPhysical>) -> PolarsResult<ChunkedArray<T::PolarsPhysical>>,
    {
        let cats = apply(self.0.physical())?;
        unsafe {
            Ok(CategoricalChunked::from_cats_and_dtype_unchecked(
                cats,
                self.0.dtype().clone(),
            ))
        }
    }
}

macro_rules! impl_cat_series {
    ($ca: ident, $pdt:ty) => {
        impl private::PrivateSeries for SeriesWrap<$ca> {
            fn compute_len(&mut self) {
                self.0.physical_mut().compute_len()
            }
            fn _field(&self) -> Cow<'_, Field> {
                Cow::Owned(self.0.field())
            }
            fn _dtype(&self) -> &DataType {
                self.0.dtype()
            }
            fn _get_flags(&self) -> StatisticsFlags {
                self.0.get_flags()
            }
            fn _set_flags(&mut self, flags: StatisticsFlags) {
                self.0.set_flags(flags)
            }

            unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
                self.0.physical().equal_element(idx_self, idx_other, other)
            }

            #[cfg(feature = "zip_with")]
            fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
                polars_ensure!(self.dtype() == other.dtype(), SchemaMismatch: "expected '{}' found '{}'", self.dtype(), other.dtype());
                let other = other.to_physical_repr().into_owned();
                unsafe {
                    Ok(self.try_apply_on_phys(|ca| {
                        ca.zip_with(mask, other.as_ref().as_ref())
                    })?.into_series())
                }
            }

            fn into_total_ord_inner<'a>(&'a self) -> Box<dyn TotalOrdInner + 'a> {
                if self.0.uses_lexical_ordering() {
                    (&self.0).into_total_ord_inner()
                } else {
                    self.0.physical().into_total_ord_inner()
                }
            }
            fn into_total_eq_inner<'a>(&'a self) -> Box<dyn TotalEqInner + 'a> {
                invalid_operation_panic!(into_total_eq_inner, self)
            }

            fn vec_hash(
                &self,
                random_state: PlSeedableRandomStateQuality,
                buf: &mut Vec<u64>,
            ) -> PolarsResult<()> {
                self.0.physical().vec_hash(random_state, buf)?;
                Ok(())
            }

            fn vec_hash_combine(
                &self,
                build_hasher: PlSeedableRandomStateQuality,
                hashes: &mut [u64],
            ) -> PolarsResult<()> {
                self.0.physical().vec_hash_combine(build_hasher, hashes)?;
                Ok(())
            }

            #[cfg(feature = "algorithm_group_by")]
            unsafe fn agg_list(&self, groups: &GroupsType) -> Series {
                // we cannot cast and dispatch as the inner type of the list would be incorrect
                let list = self.0.physical().agg_list(groups);
                let mut list = list.list().unwrap().clone();
                unsafe { list.to_logical(self.dtype().clone()) };
                list.into_series()
            }

            #[cfg(feature = "algorithm_group_by")]
            fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsType> {
                self.0.physical().group_tuples(multithreaded, sorted)
            }

            fn arg_sort_multiple(
                &self,
                by: &[Column],
                options: &SortMultipleOptions,
            ) -> PolarsResult<IdxCa> {
                self.0.arg_sort_multiple(by, options)
            }
        }

        impl SeriesTrait for SeriesWrap<$ca> {
            fn rename(&mut self, name: PlSmallStr) {
                self.0.physical_mut().rename(name);
            }

            fn chunk_lengths(&self) -> ChunkLenIter<'_> {
                self.0.physical().chunk_lengths()
            }

            fn name(&self) -> &PlSmallStr {
                self.0.physical().name()
            }

            fn chunks(&self) -> &Vec<ArrayRef> {
                self.0.physical().chunks()
            }

            unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
                self.0.physical_mut().chunks_mut()
            }

            fn shrink_to_fit(&mut self) {
                self.0.physical_mut().shrink_to_fit()
            }

            fn slice(&self, offset: i64, length: usize) -> Series {
                unsafe { self.apply_on_phys(|cats| cats.slice(offset, length)).into_series() }
            }

            fn split_at(&self, offset: i64) -> (Series, Series) {
                unsafe {
                    let (a, b) = self.0.physical().split_at(offset);
                    let a = <$ca>::from_cats_and_dtype_unchecked(a, self.0.dtype().clone()).into_series();
                    let b = <$ca>::from_cats_and_dtype_unchecked(b, self.0.dtype().clone()).into_series();
                    (a, b)
                }
            }

            fn append(&mut self, other: &Series) -> PolarsResult<()> {
                polars_ensure!(self.0.dtype() == other.dtype(), append);
                self.0.append(other.cat::<$pdt>().unwrap())
            }

            fn append_owned(&mut self, mut other: Series) -> PolarsResult<()> {
                polars_ensure!(self.0.dtype() == other.dtype(), append);
                self.0.physical_mut().append_owned(std::mem::take(
                    other
                        ._get_inner_mut()
                        .as_any_mut()
                        .downcast_mut::<$ca>()
                        .unwrap()
                        .physical_mut(),
                ))
            }

            fn extend(&mut self, other: &Series) -> PolarsResult<()> {
                polars_ensure!(self.0.dtype() == other.dtype(), extend);
                self.0.extend(other.cat::<$pdt>().unwrap())
            }

            fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
                unsafe { Ok(self.try_apply_on_phys(|cats| cats.filter(filter))?.into_series()) }
            }

            fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
                unsafe { Ok(self.try_apply_on_phys(|cats| cats.take(indices))?.into_series() ) }
            }

            unsafe fn take_unchecked(&self, indices: &IdxCa) -> Series {
                unsafe { self.apply_on_phys(|cats| cats.take_unchecked(indices)).into_series() }
            }

            fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Series> {
                unsafe { Ok(self.try_apply_on_phys(|cats| cats.take(indices))?.into_series()) }
            }

            unsafe fn take_slice_unchecked(&self, indices: &[IdxSize]) -> Series {
                unsafe { self.apply_on_phys(|cats| cats.take_unchecked(indices)).into_series() }
            }

            fn len(&self) -> usize {
                self.0.len()
            }

            fn rechunk(&self) -> Series {
                unsafe { self.apply_on_phys(|cats| cats.rechunk().into_owned()).into_series() }
            }

            fn new_from_index(&self, index: usize, length: usize) -> Series {
                unsafe { self.apply_on_phys(|cats| cats.new_from_index(index, length)).into_series() }
            }

            fn cast(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
                self.0.cast_with_options(dtype, options)
            }

            #[inline]
            unsafe fn get_unchecked(&self, index: usize) -> AnyValue<'_> {
                self.0.get_any_value_unchecked(index)
            }

            fn sort_with(&self, options: SortOptions) -> PolarsResult<Series> {
                Ok(self.0.sort_with(options).into_series())
            }

            fn arg_sort(&self, options: SortOptions) -> IdxCa {
                self.0.arg_sort(options)
            }

            fn null_count(&self) -> usize {
                self.0.physical().null_count()
            }

            fn has_nulls(&self) -> bool {
                self.0.physical().has_nulls()
            }

            #[cfg(feature = "algorithm_group_by")]
            fn unique(&self) -> PolarsResult<Series> {
                unsafe { Ok(self.try_apply_on_phys(|cats| cats.unique())?.into_series()) }
            }

            #[cfg(feature = "algorithm_group_by")]
            fn n_unique(&self) -> PolarsResult<usize> {
                self.0.physical().n_unique()
            }

            #[cfg(feature = "algorithm_group_by")]
            fn arg_unique(&self) -> PolarsResult<IdxCa> {
                self.0.physical().arg_unique()
            }

            fn is_null(&self) -> BooleanChunked {
                self.0.physical().is_null()
            }

            fn is_not_null(&self) -> BooleanChunked {
                self.0.physical().is_not_null()
            }

            fn reverse(&self) -> Series {
                unsafe { self.apply_on_phys(|cats| cats.reverse()).into_series() }
            }

            fn as_single_ptr(&mut self) -> PolarsResult<usize> {
                self.0.physical_mut().as_single_ptr()
            }

            fn shift(&self, periods: i64) -> Series {
                unsafe { self.apply_on_phys(|ca| ca.shift(periods)).into_series() }
            }

            fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
                Arc::new(SeriesWrap(Clone::clone(&self.0)))
            }

            fn min_reduce(&self) -> PolarsResult<Scalar> {
                Ok(ChunkAggSeries::min_reduce(&self.0))
            }

            fn max_reduce(&self) -> PolarsResult<Scalar> {
                Ok(ChunkAggSeries::max_reduce(&self.0))
            }

            fn find_validity_mismatch(&self, other: &Series, idxs: &mut Vec<IdxSize>) {
                self.0.physical().find_validity_mismatch(other, idxs)
            }

            fn as_any(&self) -> &dyn Any {
                &self.0
            }

            fn as_any_mut(&mut self) -> &mut dyn Any {
                &mut self.0
            }

            fn as_phys_any(&self) -> &dyn Any {
                self.0.physical()
            }

            fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
                self as _
            }
        }

        impl private::PrivateSeriesNumeric for SeriesWrap<$ca> {
            fn bit_repr(&self) -> Option<BitRepr> {
                Some(self.0.physical().to_bit_repr())
            }
        }
    }
}

impl_cat_series!(Categorical8Chunked, Categorical8Type);
impl_cat_series!(Categorical16Chunked, Categorical16Type);
impl_cat_series!(Categorical32Chunked, Categorical32Type);
