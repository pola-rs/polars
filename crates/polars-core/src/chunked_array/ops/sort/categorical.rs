use num_traits::Zero;

use super::*;

impl<T: PolarsCategoricalType> CategoricalChunked<T> {
    #[must_use]
    pub fn sort_with(&self, options: SortOptions) -> CategoricalChunked<T> {
        if !self.uses_lexical_ordering() {
            let cats = self.physical().sort_with(options);
            // SAFETY: we only reordered the indexes so we are still in bounds.
            return unsafe {
                CategoricalChunked::<T>::from_cats_and_dtype_unchecked(cats, self.dtype().clone())
            };
        }

        let mut vals = self
            .physical()
            .into_iter()
            .zip(self.iter_str())
            .collect_trusted::<Vec<_>>();

        sort_unstable_by_branch(vals.as_mut_slice(), options, |a, b| a.1.cmp(&b.1));

        let mut cats = Vec::with_capacity(self.len());
        let mut validity =
            (self.null_count() > 0).then(|| BitmapBuilder::with_capacity(self.len()));

        if self.null_count() > 0 && !options.nulls_last {
            cats.resize(self.null_count(), T::Native::zero());
            if let Some(validity) = &mut validity {
                validity.extend_constant(self.null_count(), false);
            }
        }

        let valid_slice = if options.descending {
            &vals[..self.len() - self.null_count()]
        } else {
            &vals[self.null_count()..]
        };
        cats.extend(valid_slice.iter().map(|(idx, _v)| idx.unwrap()));
        if let Some(validity) = &mut validity {
            validity.extend_constant(self.len() - self.null_count(), true);
        }

        if self.null_count() > 0 && options.nulls_last {
            cats.resize(self.len(), T::Native::zero());
            if let Some(validity) = &mut validity {
                validity.extend_constant(self.null_count(), false);
            }
        }

        let arr = PrimitiveArray::from_vec(cats).with_validity(validity.map(|v| v.freeze()));
        let cats = ChunkedArray::with_chunk(self.name().clone(), arr);

        // SAFETY: we only reordered the indexes so we are still in bounds.
        unsafe {
            CategoricalChunked::<T>::from_cats_and_dtype_unchecked(cats, self.dtype().clone())
        }
    }

    /// Returned a sorted `ChunkedArray`.
    #[must_use]
    pub fn sort(&self, descending: bool) -> CategoricalChunked<T> {
        self.sort_with(SortOptions {
            nulls_last: false,
            descending,
            multithreaded: true,
            maintain_order: false,
            limit: None,
        })
    }

    /// Retrieve the indexes needed to sort this array.
    pub fn arg_sort(&self, options: SortOptions) -> IdxCa {
        if self.uses_lexical_ordering() {
            let iters = [self.iter_str()];
            arg_sort::arg_sort(
                self.name().clone(),
                iters,
                options,
                self.physical().null_count(),
                self.len(),
                IsSorted::Not,
                false,
            )
        } else {
            self.physical().arg_sort(options)
        }
    }

    /// Retrieve the indices needed to sort this and the other arrays.
    pub(crate) fn arg_sort_multiple(
        &self,
        by: &[Column],
        options: &SortMultipleOptions,
    ) -> PolarsResult<IdxCa> {
        if self.uses_lexical_ordering() {
            args_validate(self.physical(), by, &options.descending, "descending")?;
            args_validate(self.physical(), by, &options.nulls_last, "nulls_last")?;
            let mut count: IdxSize = 0;

            // we use bytes to save a monomorphisized str impl
            // as bytes already is used for binary and string sorting
            let vals: Vec<_> = self
                .iter_str()
                .map(|v| {
                    let i = count;
                    count += 1;
                    (i, v.map(|v| v.as_bytes()))
                })
                .collect_trusted();

            arg_sort_multiple_impl(vals, by, options)
        } else {
            self.physical().arg_sort_multiple(by, options)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    fn assert_order(ca: &Categorical8Chunked, cmp: &[&str]) {
        let s = ca.cast(&DataType::String).unwrap();
        let ca = s.str().unwrap();
        assert_eq!(ca.into_no_null_iter().collect::<Vec<_>>(), cmp);
    }

    #[test]
    fn test_cat_lexical_sort() -> PolarsResult<()> {
        let init = &["c", "b", "a", "d"];

        let cats = Categories::new(
            PlSmallStr::EMPTY,
            PlSmallStr::EMPTY,
            CategoricalPhysical::U8,
        );
        let s = Series::new(PlSmallStr::EMPTY, init).cast(&DataType::from_categories(cats))?;
        let ca = s.cat8()?;

        let out = ca.sort(false);
        assert_order(&out, &["a", "b", "c", "d"]);

        let out = ca.arg_sort(SortOptions {
            descending: false,
            ..Default::default()
        });
        assert_eq!(out.into_no_null_iter().collect::<Vec<_>>(), &[2, 1, 0, 3]);

        Ok(())
    }

    #[test]
    fn test_cat_lexical_sort_multiple() -> PolarsResult<()> {
        let init = &["c", "b", "a", "a"];

        let cats = Categories::new(
            PlSmallStr::EMPTY,
            PlSmallStr::EMPTY,
            CategoricalPhysical::U8,
        );
        let series = Series::new(PlSmallStr::EMPTY, init).cast(&DataType::from_categories(cats))?;

        let df = df![
            "cat" => &series,
            "vals" => [1, 1, 2, 2]
        ]?;

        let out = df.sort(
            ["cat", "vals"],
            SortMultipleOptions::default().with_order_descending_multi([false, false]),
        )?;
        let out = out.column("cat")?;
        let cat = out.as_materialized_series().cat8()?;
        assert_order(cat, &["a", "a", "b", "c"]);

        let out = df.sort(
            ["vals", "cat"],
            SortMultipleOptions::default().with_order_descending_multi([false, false]),
        )?;
        let out = out.column("cat")?;
        let cat = out.as_materialized_series().cat8()?;
        assert_order(cat, &["b", "c", "a", "a"]);

        Ok(())
    }
}
