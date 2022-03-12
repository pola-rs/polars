use super::*;
use crate::utils::NoNull;

/// Default sorting nulls
pub fn order_default_null<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    sort_with_nulls(a, b)
}

/// Default sorting nulls
pub fn order_reverse_null<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    sort_with_nulls(b, a)
}

impl CategoricalChunked {
    #[must_use]
    pub fn sort_with(&self, options: SortOptions) -> CategoricalChunked {
        assert!(
            !options.nulls_last,
            "null last not yet supported for categorical dtype"
        );

        if self.use_lexical_sort() {
            match &**self.get_rev_map() {
                RevMapping::Local(arr) => {
                    // we don't use arrow2 sort here because its not activated
                    // that saves compilation
                    let ca = Utf8Chunked::from_chunks("", vec![Arc::from(arr.clone())]);
                    let sorted = ca.sort(options.descending);
                    let arr = sorted.downcast_iter().next().unwrap().clone();
                    let rev_map = RevMapping::Local(arr);
                    CategoricalChunked::from_cats_and_rev_map(
                        self.logical().clone(),
                        Arc::new(rev_map),
                    )
                }
                RevMapping::Global(_, _, _) => {
                    // a global rev map must always point to the same string values
                    // so we cannot sort the categories.

                    let mut vals = self
                        .logical()
                        .into_no_null_iter()
                        .zip(self.iter_str())
                        .collect_trusted::<Vec<_>>();

                    argsort_branch(
                        vals.as_mut_slice(),
                        options.descending,
                        |(_, a), (_, b)| order_default_null(a, b),
                        |(_, a), (_, b)| order_reverse_null(a, b),
                    );
                    let cats: NoNull<UInt32Chunked> =
                        vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
                    CategoricalChunked::from_cats_and_rev_map(
                        cats.into_inner(),
                        self.get_rev_map().clone(),
                    )
                }
            }
        } else {
            let cats = self.logical().sort_with(options);
            CategoricalChunked::from_cats_and_rev_map(cats, self.get_rev_map().clone())
        }
    }

    /// Returned a sorted `ChunkedArray`.
    #[must_use]
    pub fn sort(&self, reverse: bool) -> CategoricalChunked {
        self.sort_with(SortOptions {
            nulls_last: false,
            descending: reverse,
        })
    }

    /// Retrieve the indexes needed to sort this array.
    pub fn argsort(&self, options: SortOptions) -> IdxCa {
        if self.use_lexical_sort() {
            let iters = [self.iter_str()];
            argsort::argsort(
                self.name(),
                iters,
                options,
                self.logical().null_count(),
                self.len(),
            )
        } else {
            self.logical().argsort(options)
        }
    }

    /// Retrieve the indexes need to sort this and the other arrays.
    pub(crate) fn argsort_multiple(&self, other: &[Series], reverse: &[bool]) -> Result<IdxCa> {
        if self.use_lexical_sort() {
            args_validate(self.logical(), other, reverse)?;
            let mut count: IdxSize = 0;
            let vals: Vec<_> = self
                .iter_str()
                .map(|v| {
                    let i = count;
                    count += 1;
                    (i, v)
                })
                .collect_trusted();

            argsort_multiple_impl(vals, other, reverse)
        } else {
            self.logical().argsort_multiple(other, reverse)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::{toggle_string_cache, SINGLE_LOCK};

    fn assert_order(ca: &CategoricalChunked, cmp: &[&str]) {
        let s = ca.cast(&DataType::Utf8).unwrap();
        let ca = s.utf8().unwrap();
        assert_eq!(ca.into_no_null_iter().collect::<Vec<_>>(), cmp);
    }

    #[test]
    fn test_cat_lexical_sort() -> Result<()> {
        let init = &["c", "b", "a", "d"];

        let _lock = SINGLE_LOCK.lock();
        for toggle in [true, false] {
            toggle_string_cache(toggle);
            let s = Series::new("", init).cast(&DataType::Categorical(None))?;
            let ca = s.categorical()?;
            let mut ca_lexical = ca.clone();
            ca_lexical.set_lexical_sorted(true);

            let out = ca_lexical.sort(false);
            assert_order(&out, &["a", "b", "c", "d"]);
            let out = ca.sort(false);
            assert_order(&out, init);

            let out = ca_lexical.argsort(SortOptions {
                descending: false,
                ..Default::default()
            });
            assert_eq!(out.into_no_null_iter().collect::<Vec<_>>(), &[2, 1, 0, 3]);
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "sort_multiple")]
    fn test_cat_lexical_sort_multiple() -> Result<()> {
        let init = &["c", "b", "a", "a"];

        let _lock = SINGLE_LOCK.lock();
        for toggle in [true, false] {
            toggle_string_cache(toggle);
            let s = Series::new("", init).cast(&DataType::Categorical(None))?;
            let ca = s.categorical()?;
            let mut ca_lexical = ca.clone();
            ca_lexical.set_lexical_sorted(true);

            let df = df![
                "cat" => &ca_lexical.into_series(),
                "vals" => [1, 1, 2, 2]
            ]?;

            let out = df.sort(&["cat", "vals"], vec![false, false])?;
            let out = out.column("cat")?;
            let cat = out.categorical()?;
            assert_order(cat, &["a", "a", "b", "c"]);

            let out = df.sort(&["vals", "cat"], vec![false, false])?;
            let out = out.column("cat")?;
            let cat = out.categorical()?;
            assert_order(cat, &["b", "c", "a", "a"]);
        }
        Ok(())
    }
}
