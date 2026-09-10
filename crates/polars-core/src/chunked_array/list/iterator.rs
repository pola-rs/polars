use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::chunked_array::flags::StatisticsFlags;
use crate::chunked_array::list::collect_list_chunk;
use crate::prelude::*;
use crate::series::amortized_iter::{AmortSeries, ArrayBox, unstable_series_container_and_ptr};

pub struct AmortizedListIter<'a, I: Iterator<Item = Option<ArrayBox>>> {
    len: usize,
    series_container: Rc<Series>,
    inner: NonNull<PlArrayRef>,
    lifetime: PhantomData<&'a PlArrayRef>,
    iter: I,
    // used only if feature="dtype-struct"
    #[allow(dead_code)]
    inner_dtype: DataType,
}

impl<I: Iterator<Item = Option<ArrayBox>>> AmortizedListIter<'_, I> {
    pub(crate) unsafe fn new(
        len: usize,
        series_container: Series,
        inner: NonNull<PlArrayRef>,
        iter: I,
        inner_dtype: DataType,
    ) -> Self {
        Self {
            len,
            series_container: Rc::new(series_container),
            inner,
            lifetime: PhantomData,
            iter,
            inner_dtype,
        }
    }
}

impl<I: Iterator<Item = Option<ArrayBox>>> Iterator for AmortizedListIter<'_, I> {
    type Item = Option<AmortSeries>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|opt_val| {
            opt_val.map(|array_ref| {
                #[cfg(feature = "dtype-struct")]
                // structs arrays are bound to the series not to the arrayref
                // so we must get a hold to the new array
                if matches!(self.inner_dtype, DataType::Struct(_)) {
                    // SAFETY:
                    // dtype is known
                    unsafe {
                        let s = Series::from_chunks_and_dtype_unchecked(
                            self.series_container.name().clone(),
                            vec![array_ref],
                            &self.inner_dtype.to_physical(),
                        )
                        .from_physical_unchecked(&self.inner_dtype)
                        .unwrap();
                        let inner = Rc::make_mut(&mut self.series_container);
                        *inner = s;

                        return AmortSeries::new(self.series_container.clone());
                    }
                }
                // The series is cloned, we make a new container.
                if Arc::strong_count(&self.series_container.0) > 1
                    || Rc::strong_count(&self.series_container) > 1
                {
                    let (s, ptr) = unsafe {
                        unstable_series_container_and_ptr(
                            self.series_container.name().clone(),
                            array_ref,
                            self.series_container.dtype(),
                        )
                    };
                    self.series_container = Rc::new(s);
                    self.inner = NonNull::new(ptr).unwrap();
                } else {
                    // SAFETY: we checked the RC above;
                    let series_mut =
                        unsafe { Rc::get_mut(&mut self.series_container).unwrap_unchecked() };
                    // update the inner state
                    unsafe { *self.inner.as_mut() = array_ref };

                    // As an optimization, we try to minimize how many calls to
                    // _get_inner_mut() we do.
                    let series_mut_inner = series_mut._get_inner_mut();
                    // last iteration could have set the sorted flag (e.g. in compute_len)
                    series_mut_inner._set_flags(StatisticsFlags::empty());
                    // make sure that the length is correct
                    series_mut_inner.compute_len();
                }

                AmortSeries::new(self.series_container.clone())
            })
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

// # Safety
// we correctly implemented size_hint
unsafe impl<I: Iterator<Item = Option<ArrayBox>>> TrustedLen for AmortizedListIter<'_, I> {}
impl<I: Iterator<Item = Option<ArrayBox>>> ExactSizeIterator for AmortizedListIter<'_, I> {}

impl ListChunked {
    /// This is an iterator over a [`ListChunked`] that saves allocations.
    /// A Series is:
    ///     1. [`Arc<ChunkedArray>`]
    ///     ChunkedArray is:
    ///         2. Vec< 3. ArrayRef>
    ///
    /// The ArrayRef we indicated with 3. will be updated during iteration.
    /// The Series will be pinned in memory, saving an allocation for
    /// 1. Arc<..>
    /// 2. Vec<...>
    ///
    /// If the returned `AmortSeries` is cloned, the local copy will be replaced and a new container
    /// will be set.
    pub fn amortized_iter(
        &self,
    ) -> AmortizedListIter<'_, impl Iterator<Item = Option<ArrayBox>> + '_> {
        self.amortized_iter_with_name(PlSmallStr::EMPTY)
    }

    /// See `amortized_iter`.
    pub fn amortized_iter_with_name(
        &self,
        name: PlSmallStr,
    ) -> AmortizedListIter<'_, impl Iterator<Item = Option<ArrayBox>> + '_> {
        // we create the series container from the inner array
        // so that the container has the proper dtype.
        let arr = self.downcast_iter().next().unwrap();
        let inner_values = arr.values().to_boxed();

        let inner_dtype = self.inner_dtype();
        let iter_dtype = match inner_dtype {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => inner_dtype.to_physical(),
            // TODO: figure out how to deal with physical/logical distinction
            // physical primitives like time, date etc. work
            // physical nested need more
            _ => inner_dtype.clone(),
        };

        // SAFETY:
        // inner type passed as physical type
        let (s, ptr) =
            unsafe { unstable_series_container_and_ptr(name, inner_values, &iter_dtype) };

        // SAFETY: ptr belongs the Series..
        unsafe {
            AmortizedListIter::new(
                self.len(),
                s,
                NonNull::new(ptr).unwrap(),
                self.downcast_iter().flat_map(|arr| arr.iter()),
                inner_dtype.clone(),
            )
        }
    }

    /// The number of elements every one of which reads the one list this array holds, if that is
    /// how it holds them: a single chunk whose offsets repeat one range, with no null element to
    /// read anything else for.
    ///
    /// A closure applied element by element then only has to see that one list: its answer is the
    /// answer for every element, and the array of them repeats it rather than holding a copy per
    /// element. See [`apply_amortized_same_type`](Self::apply_amortized_same_type).
    ///
    /// Only work that answers the same way twice may be shared like this: an unseeded sample has
    /// to be taken per element even here.
    pub fn repeats_one_list(&self) -> Option<usize> {
        let [chunk] = self.chunks().as_slice() else {
            return None;
        };
        let arr = chunk.as_any().downcast_ref::<PlListArray>()?;

        (arr.len() > 1 && arr.null_count() == 0 && arr.scalar_offsets().is_some())
            .then_some(arr.len())
    }

    /// `length` elements over the single list `out`, which every one of them reads.
    ///
    /// The values hold that list once, however many elements repeat it, and the inner type stays
    /// the one this array already names — the caller having applied a closure that keeps it.
    fn repeat_one_answer(&self, out: &Series, length: usize) -> Self {
        let arr = PlListArray::new_scalar(to_arr(out), length);
        let mut ca = ChunkedArray::from_chunk_iter_and_field(self.field.clone(), [arr]);

        // Every element reads the one list, so they are all empty or none of them is.
        if !out.is_empty() {
            ca.set_fast_explode();
        }
        ca
    }

    /// [`repeat_one_answer`](Self::repeat_one_answer), for a closure that changed the inner type:
    /// the answer says what is under the lists now.
    fn repeat_one_answer_of_dtype(&self, out: &Series, length: usize) -> Self {
        let arr = PlListArray::new_scalar(to_arr(out), length);

        // SAFETY: the values are the answer itself, so the inner type is the one it carries.
        let mut ca = unsafe {
            ListChunked::from_chunks_and_dtype_unchecked(
                self.name().clone(),
                vec![Box::new(arr)],
                DataType::List(Box::new(out.dtype().clone())),
            )
        };

        if !out.is_empty() {
            ca.set_fast_explode();
        }
        ca
    }

    /// The one list every element reads, for [`repeats_one_list`](Self::repeats_one_list) to have
    /// answered `Some`.
    fn one_list(&self) -> AmortSeries {
        self.amortized_iter()
            .next()
            .flatten()
            .expect("an array that repeats one list holds it, and holds it for every element")
    }

    /// Applies a closure `F` elementwise.
    #[must_use]
    pub fn apply_amortized_generic<F, K, V>(&self, f: F) -> ChunkedArray<V>
    where
        V: PolarsDataType,
        F: FnMut(Option<AmortSeries>) -> Option<K> + Copy,
        V::Array: ArrayFromIter<Option<K>>,
    {
        // TODO! make an amortized iter that does not flatten
        self.amortized_iter().map(f).collect_ca(self.name().clone())
    }

    pub fn try_apply_amortized_generic<F, K, V>(&self, f: F) -> PolarsResult<ChunkedArray<V>>
    where
        V: PolarsDataType,
        F: FnMut(Option<AmortSeries>) -> PolarsResult<Option<K>> + Copy,
        V::Array: ArrayFromIter<Option<K>>,
    {
        // TODO! make an amortized iter that does not flatten
        self.amortized_iter()
            .map(f)
            .try_collect_ca(self.name().clone())
    }

    pub fn for_each_amortized<F>(&self, f: F)
    where
        F: FnMut(Option<AmortSeries>),
    {
        self.amortized_iter().for_each(f)
    }

    /// Zip with a `ChunkedArray` then apply a binary function `F` elementwise.
    #[must_use]
    pub fn zip_and_apply_amortized<'a, T, F>(&'a self, ca: &'a ChunkedArray<T>, mut f: F) -> Self
    where
        T: PolarsDataType,
        F: FnMut(Option<AmortSeries>, Option<T::Physical<'a>>) -> Option<Series>,
    {
        if self.is_empty() {
            return self.clone();
        }
        let mut fast_explode = self.null_count() == 0;
        let mut out: ListChunked = {
            self.amortized_iter()
                .zip(ca.iter())
                .map(|(opt_s, opt_v)| {
                    let out = f(opt_s, opt_v);
                    match out {
                        Some(out) => {
                            fast_explode &= !out.is_empty();
                            Some(out)
                        },
                        None => {
                            fast_explode = false;
                            out
                        },
                    }
                })
                .collect_trusted()
        };

        out.rename(self.name().clone());
        if fast_explode {
            out.set_fast_explode();
        }
        out
    }

    #[must_use]
    pub fn binary_zip_and_apply_amortized<'a, T, U, F>(
        &'a self,
        ca1: &'a ChunkedArray<T>,
        ca2: &'a ChunkedArray<U>,
        mut f: F,
    ) -> Self
    where
        T: PolarsDataType,
        U: PolarsDataType,
        F: FnMut(
            Option<AmortSeries>,
            Option<T::Physical<'a>>,
            Option<U::Physical<'a>>,
        ) -> Option<Series>,
    {
        if self.is_empty() {
            return self.clone();
        }
        let mut fast_explode = self.null_count() == 0;
        let mut out: ListChunked = {
            self.amortized_iter()
                .zip(ca1.iter())
                .zip(ca2.iter())
                .map(|((opt_s, opt_u), opt_v)| {
                    let out = f(opt_s, opt_u, opt_v);
                    match out {
                        Some(out) => {
                            fast_explode &= !out.is_empty();
                            Some(out)
                        },
                        None => {
                            fast_explode = false;
                            out
                        },
                    }
                })
                .collect_trusted()
        };

        out.rename(self.name().clone());
        if fast_explode {
            out.set_fast_explode();
        }
        out
    }

    pub fn try_binary_zip_and_apply_amortized<'a, T, U, F>(
        &'a self,
        ca1: &'a ChunkedArray<T>,
        ca2: &'a ChunkedArray<U>,
        mut f: F,
    ) -> PolarsResult<Self>
    where
        T: PolarsDataType,
        U: PolarsDataType,
        F: FnMut(
            Option<AmortSeries>,
            Option<T::Physical<'a>>,
            Option<U::Physical<'a>>,
        ) -> PolarsResult<Option<Series>>,
    {
        if self.is_empty() {
            return Ok(self.clone());
        }
        let mut fast_explode = self.null_count() == 0;
        let mut out: ListChunked = {
            self.amortized_iter()
                .zip(ca1.iter())
                .zip(ca2.iter())
                .map(|((opt_s, opt_u), opt_v)| {
                    let out = f(opt_s, opt_u, opt_v)?;
                    match out {
                        Some(out) => {
                            fast_explode &= !out.is_empty();
                            Ok(Some(out))
                        },
                        None => {
                            fast_explode = false;
                            Ok(out)
                        },
                    }
                })
                .collect::<PolarsResult<_>>()?
        };

        out.rename(self.name().clone());
        if fast_explode {
            out.set_fast_explode();
        }
        Ok(out)
    }

    pub fn try_zip_and_apply_amortized<'a, T, F>(
        &'a self,
        ca: &'a ChunkedArray<T>,
        mut f: F,
    ) -> PolarsResult<Self>
    where
        T: PolarsDataType,
        F: FnMut(Option<AmortSeries>, Option<T::Physical<'a>>) -> PolarsResult<Option<Series>>,
    {
        if self.is_empty() {
            return Ok(self.clone());
        }
        let mut fast_explode = self.null_count() == 0;
        let mut out: ListChunked = {
            self.amortized_iter()
                .zip(ca.iter())
                .map(|(opt_s, opt_v)| {
                    let out = f(opt_s, opt_v)?;
                    match out {
                        Some(out) => {
                            fast_explode &= !out.is_empty();
                            Ok(Some(out))
                        },
                        None => {
                            fast_explode = false;
                            Ok(out)
                        },
                    }
                })
                .collect::<PolarsResult<_>>()?
        };

        out.rename(self.name().clone());
        if fast_explode {
            out.set_fast_explode();
        }
        Ok(out)
    }

    /// Apply a closure `F` to each list elementwise.
    ///
    /// # Safety
    /// The closure `F` must return the same dtype as the input.
    #[must_use]
    pub unsafe fn apply_amortized_same_type<F>(&self, mut f: F) -> Self
    where
        F: FnMut(AmortSeries) -> Series,
    {
        if self.is_empty() {
            return self.clone();
        }

        // The one list every element reads is mapped once, and the answer is that one list
        // repeated: `f` runs once rather than once per element, and the elements share it.
        if let Some(length) = self.repeats_one_list() {
            let out = f(self.one_list());
            return self.repeat_one_answer(&out, length);
        }

        let mut fast_explode = self.null_count() == 0;
        let elements = self
            .amortized_iter()
            .map(|opt_v| {
                opt_v.map(|v| {
                    let out = f(v);
                    if out.is_empty() {
                        fast_explode = false;
                    }
                    to_arr(&out)
                })
            })
            .collect::<Vec<_>>();
        let chunk = collect_list_chunk(elements, self.inner_dtype());
        let mut ca = ChunkedArray::from_chunk_iter_and_field(self.field.clone(), [chunk]);

        if fast_explode {
            ca.set_fast_explode();
        }
        ca
    }

    /// Try apply a closure `F` elementwise (may change dtype).
    pub fn try_apply_amortized<F>(&self, mut f: F) -> PolarsResult<Self>
    where
        F: FnMut(AmortSeries) -> PolarsResult<Series>,
    {
        if self.is_empty() {
            return Ok(self.clone());
        }

        // As in `apply_amortized_same_type`, with the dtype the one answer came back as.
        if let Some(length) = self.repeats_one_list() {
            let out = f(self.one_list())?;
            return Ok(self.repeat_one_answer_of_dtype(&out, length));
        }

        let mut fast_explode = self.null_count() == 0;
        let mut ca: ListChunked = {
            self.amortized_iter()
                .map(|opt_v| {
                    opt_v
                        .map(|v| {
                            let out = f(v);
                            if let Ok(out) = &out {
                                if out.is_empty() {
                                    fast_explode = false
                                }
                            };
                            out
                        })
                        .transpose()
                })
                .collect::<PolarsResult<_>>()?
        };
        ca.rename(self.name().clone());
        if fast_explode {
            ca.set_fast_explode();
        }
        Ok(ca)
    }

    /// Try apply a closure `F` to each list element.
    ///
    /// # Safety
    /// The closure `F` must return the same dtype as the input.
    pub unsafe fn try_apply_amortized_same_type<F>(&self, mut f: F) -> PolarsResult<Self>
    where
        F: FnMut(AmortSeries) -> PolarsResult<Series>,
    {
        // As in `apply_amortized_same_type`: one list mapped once, the answer shared.
        if !self.is_empty()
            && let Some(length) = self.repeats_one_list()
        {
            let out = f(self.one_list())?;
            return Ok(self.repeat_one_answer(&out, length));
        }

        // SAFETY: the caller's guarantee is the one this asks for.
        unsafe { self.try_apply_amortized_same_type_per_element(f) }
    }

    /// [`try_apply_amortized_same_type`](Self::try_apply_amortized_same_type), applying `f` to
    /// every element even where they all read the one list.
    ///
    /// This is what an `f` that answers differently on the same list from one call to the next
    /// asks for — one that samples without a seed, say. An `f` that does not should take the
    /// method above, which then only calls it once.
    ///
    /// # Safety
    /// The closure `F` must return the same dtype as the input.
    pub unsafe fn try_apply_amortized_same_type_per_element<F>(
        &self,
        mut f: F,
    ) -> PolarsResult<Self>
    where
        F: FnMut(AmortSeries) -> PolarsResult<Series>,
    {
        if self.is_empty() {
            return Ok(self.clone());
        }
        let mut fast_explode = self.null_count() == 0;
        let elements = self
            .amortized_iter()
            .map(|opt_v| {
                opt_v
                    .map(|v| {
                        let out = f(v)?;
                        if out.is_empty() {
                            fast_explode = false;
                        }
                        PolarsResult::Ok(to_arr(&out))
                    })
                    .transpose()
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        let chunk = collect_list_chunk(elements, self.inner_dtype());
        let mut ca = ChunkedArray::from_chunk_iter_and_field(self.field.clone(), [chunk]);

        if fast_explode {
            ca.set_fast_explode();
        }
        Ok(ca)
    }
}

fn to_arr(s: &Series) -> PlArrayRef {
    if s.chunks().len() > 1 {
        let s = s.rechunk();
        s.chunks()[0].clone()
    } else {
        s.chunks()[0].clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::chunked_array::builder::get_list_builder;

    #[test]
    fn test_iter_list() {
        let mut builder = get_list_builder(&DataType::Int32, 10, 10, PlSmallStr::EMPTY);
        builder
            .append_series(&Series::new(PlSmallStr::EMPTY, &[1, 2, 3]))
            .unwrap();
        builder
            .append_series(&Series::new(PlSmallStr::EMPTY, &[3, 2, 1]))
            .unwrap();
        builder
            .append_series(&Series::new(PlSmallStr::EMPTY, &[1, 1]))
            .unwrap();
        let ca = builder.finish();

        ca.amortized_iter()
            .zip(ca.series_iter())
            .for_each(|(s1, s2)| {
                assert!(s1.unwrap().as_ref().equals(&s2.unwrap()));
            })
    }
}
