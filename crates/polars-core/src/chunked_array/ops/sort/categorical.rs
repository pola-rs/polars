use num_traits::Zero;

use super::*;

impl<T: PolarsCategoricalType> CategoricalChunked<T> {
    /// Returns `(to_rank, from_rank)`, ranking only the categories present in `self` so
    /// this stays cheap even when the shared `Categories` mapping holds far more
    /// categories than this column uses. `from_rank` is the inverse of `to_rank`.
    fn lexical_rank_mapping(&self) -> (Vec<T::Native>, Vec<T::Native>) {
        let mapping = self.get_mapping();
        let upper_bound = mapping.num_cats_upper_bound();

        let mut present = vec![false; upper_bound];
        for arr in self.physical().downcast_iter() {
            for cat in arr.non_null_values_iter() {
                present[cat.as_cat() as usize] = true;
            }
        }

        let mut from_rank: Vec<T::Native> = (0..upper_bound as u32)
            .filter(|&c| present[c as usize])
            .map(T::Native::from_cat)
            .collect();
        // SAFETY: every id in `from_rank` came from a category id actually present in the
        // column, so it is guaranteed to have a string value in the mapping.
        from_rank.sort_unstable_by_key(|cat_id| unsafe {
            mapping.cat_to_str_unchecked(cat_id.as_cat())
        });

        let mut to_rank = vec![T::Native::zero(); upper_bound];
        for (rank, cat_id) in from_rank.iter().enumerate() {
            to_rank[cat_id.as_cat() as usize] = T::Native::from_cat(rank as u32);
        }

        (to_rank, from_rank)
    }

    #[must_use]
    pub fn sort_with(&self, options: SortOptions) -> CategoricalChunked<T> {
        if !self.uses_lexical_ordering() {
            let cats = self.physical().sort_with(options);
            // SAFETY: we only reordered the indexes so we are still in bounds.
            return unsafe {
                CategoricalChunked::<T>::from_cats_and_dtype_unchecked(cats, self.dtype().clone())
            };
        }

        // map -> sort -> unmap (see #28774): an ordinary integer sort on the rank
        // reproduces the lexical order without a string compare per pair.
        let (to_rank, from_rank) = self.lexical_rank_mapping();
        let ranks = self
            .physical()
            .apply(|opt_cat| opt_cat.map(|cat| to_rank[cat.as_cat() as usize]));
        let sorted_ranks = ranks.sort_with(options);
        let cats =
            sorted_ranks.apply(|opt_rank| opt_rank.map(|rank| from_rank[rank.as_cat() as usize]));

        // SAFETY: we only reordered and relabeled within the existing set of category ids.
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
            // Only the resulting index order matters here, so no need to unmap back
            // to category ids.
            let (to_rank, _) = self.lexical_rank_mapping();
            let ranks = self
                .physical()
                .apply(|opt_cat| opt_cat.map(|cat| to_rank[cat.as_cat() as usize]));
            ranks.arg_sort(options)
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

            let (to_rank, _) = self.lexical_rank_mapping();
            let vals: Vec<_> = self
                .physical()
                .iter()
                .map(|opt_cat| {
                    let i = count;
                    count += 1;
                    (i, opt_cat.map(|cat| to_rank[cat.as_cat() as usize]))
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
        assert_eq!(ca.no_null_iter().collect::<Vec<_>>(), cmp);
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

    #[test]
    fn test_cat_lexical_sort_partial_mapping() -> PolarsResult<()> {
        // The shared `Categories` mapping may hold categories this particular column never
        // uses; the ranking must be built only from what is physically present, and must
        // not be thrown off by the extra entries.
        let cats = Categories::new(
            PlSmallStr::EMPTY,
            PlSmallStr::EMPTY,
            CategoricalPhysical::U8,
        );
        let dtype = DataType::from_categories(cats);

        let _superset = Series::new(PlSmallStr::EMPTY, &["z", "y", "x", "w", "v"]).cast(&dtype)?;

        let s = Series::new(PlSmallStr::EMPTY, &["c", "b", "a", "d"]).cast(&dtype)?;
        let ca = s.cat8()?;

        let out = ca.sort(false);
        assert_order(&out, &["a", "b", "c", "d"]);

        let out = ca.sort(true);
        assert_order(&out, &["d", "c", "b", "a"]);

        Ok(())
    }

    #[test]
    fn test_cat_lexical_sort_matches_string_sort() -> PolarsResult<()> {
        // Categorical.sort() must reproduce the same order as sorting the plain strings,
        // including duplicates and nulls, across every combination of descending/nulls_last.
        let data: &[Option<&str>] = &[
            Some("banana"),
            Some("apple"),
            None,
            Some("apple"),
            Some("cherry"),
            None,
            Some("banana"),
            Some("apple"),
        ];

        let cats = Categories::new(
            PlSmallStr::EMPTY,
            PlSmallStr::EMPTY,
            CategoricalPhysical::U8,
        );
        let str_s = Series::new(PlSmallStr::EMPTY, data);
        let cat_s = str_s.cast(&DataType::from_categories(cats))?;
        let cat_ca = cat_s.cat8()?;

        for descending in [false, true] {
            for nulls_last in [false, true] {
                let opts = SortOptions {
                    descending,
                    nulls_last,
                    multithreaded: true,
                    maintain_order: false,
                    limit: None,
                };

                let str_sorted = str_s.str()?.sort_with(opts);
                let cat_sorted = cat_ca.sort_with(opts).cast(&DataType::String)?;
                assert_eq!(
                    cat_sorted.str()?.iter().collect::<Vec<_>>(),
                    str_sorted.iter().collect::<Vec<_>>(),
                    "sort_with mismatch (descending={descending}, nulls_last={nulls_last})",
                );

                let str_arg = str_s.str()?.arg_sort(opts);
                let cat_arg = cat_ca.arg_sort(opts);
                let via_str_arg = str_s.take(&str_arg)?;
                let via_cat_arg = cat_s.take(&cat_arg)?.cast(&DataType::String)?;
                assert_eq!(
                    via_cat_arg.str()?.iter().collect::<Vec<_>>(),
                    via_str_arg.str()?.iter().collect::<Vec<_>>(),
                    "arg_sort mismatch (descending={descending}, nulls_last={nulls_last})",
                );
            }
        }

        Ok(())
    }
}
