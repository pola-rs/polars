use super::*;
use crate::utils::NoNull;

/// Sort with null values, to reverse, swap the arguments.
fn sort_with_nulls<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    match (a, b) {
        (Some(a), Some(b)) => a.partial_cmp(b).unwrap(),
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// Default sorting nulls
pub fn order_ascending_null<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    sort_with_nulls(a, b)
}

/// Default sorting nulls
pub fn order_descending_null<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    sort_with_nulls(b, a)
}

impl CategoricalChunked {
    #[must_use]
    pub fn sort_with(&self, options: SortOptions) -> CategoricalChunked {
        assert!(
            !options.nulls_last,
            "null last not yet supported for categorical dtype"
        );

        if self.uses_lexical_ordering() {
            let mut vals = self
                .physical()
                .into_no_null_iter()
                .zip(self.iter_str())
                .collect_trusted::<Vec<_>>();

            arg_sort_branch(
                vals.as_mut_slice(),
                options.descending,
                |(_, a), (_, b)| order_ascending_null(a, b),
                |(_, a), (_, b)| order_descending_null(a, b),
                options.multithreaded,
            );
            let cats: NoNull<UInt32Chunked> =
                vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
            let mut cats = cats.into_inner();
            cats.rename(self.name());

            // safety:
            // we only reordered the indexes so we are still in bounds
            return unsafe {
                CategoricalChunked::from_cats_and_rev_map_unchecked(
                    cats,
                    self.get_rev_map().clone(),
                )
            };
        }
        let cats = self.physical().sort_with(options);
        // safety:
        // we only reordered the indexes so we are still in bounds
        unsafe {
            CategoricalChunked::from_cats_and_rev_map_unchecked(cats, self.get_rev_map().clone())
        }
    }

    /// Returned a sorted `ChunkedArray`.
    #[must_use]
    pub fn sort(&self, descending: bool) -> CategoricalChunked {
        self.sort_with(SortOptions {
            nulls_last: false,
            descending,
            multithreaded: true,
            maintain_order: false,
        })
    }

    /// Retrieve the indexes needed to sort this array.
    pub fn arg_sort(&self, options: SortOptions) -> IdxCa {
        if self.uses_lexical_ordering() {
            let iters = [self.iter_str()];
            arg_sort::arg_sort(
                self.name(),
                iters,
                options,
                self.physical().null_count(),
                self.len(),
            )
        } else {
            self.physical().arg_sort(options)
        }
    }

    /// Retrieve the indexes need to sort this and the other arrays.

    pub(crate) fn arg_sort_multiple(&self, options: &SortMultipleOptions) -> PolarsResult<IdxCa> {
        if self.uses_lexical_ordering() {
            args_validate(self.physical(), &options.other, &options.descending)?;
            let mut count: IdxSize = 0;

            // we use bytes to save a monomorphisized str impl
            // as bytes already is used for binary and utf8 sorting
            let vals: Vec<_> = self
                .iter_str()
                .map(|v| {
                    let i = count;
                    count += 1;
                    (i, v.map(|v| v.as_bytes()))
                })
                .collect_trusted();

            arg_sort_multiple_impl(vals, options)
        } else {
            self.physical().arg_sort_multiple(options)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::{disable_string_cache, enable_string_cache, SINGLE_LOCK};

    fn assert_order(ca: &CategoricalChunked, cmp: &[&str]) {
        let s = ca.cast(&DataType::Utf8).unwrap();
        let ca = s.utf8().unwrap();
        assert_eq!(ca.into_no_null_iter().collect::<Vec<_>>(), cmp);
    }

    #[test]
    fn test_cat_lexical_sort() -> PolarsResult<()> {
        let init = &["c", "b", "a", "d"];

        let _lock = SINGLE_LOCK.lock();
        for use_string_cache in [true, false] {
            disable_string_cache();
            if use_string_cache {
                enable_string_cache();
            }

            let s = Series::new("", init).cast(&DataType::Categorical(None))?;
            let ca = s.categorical()?;
            let mut ca_lexical = ca.clone();
            ca_lexical.set_lexical_ordering(true);

            let out = ca_lexical.sort(false);
            assert_order(&out, &["a", "b", "c", "d"]);
            let out = ca.sort(false);
            assert_order(&out, init);

            let out = ca_lexical.arg_sort(SortOptions {
                descending: false,
                ..Default::default()
            });
            assert_eq!(out.into_no_null_iter().collect::<Vec<_>>(), &[2, 1, 0, 3]);
        }

        Ok(())
    }

    #[test]
    fn test_cat_lexical_sort_multiple() -> PolarsResult<()> {
        let init = &["c", "b", "a", "a"];

        let _lock = SINGLE_LOCK.lock();
        for use_string_cache in [true, false] {
            disable_string_cache();
            if use_string_cache {
                enable_string_cache();
            }

            let s = Series::new("", init).cast(&DataType::Categorical(None))?;
            let ca = s.categorical()?;
            let mut ca_lexical: CategoricalChunked = ca.clone();
            ca_lexical.set_lexical_ordering(true);

            let series = ca_lexical.into_series();

            let df = df![
                "cat" => &series,
                "vals" => [1, 1, 2, 2]
            ]?;

            let out = df.sort(["cat", "vals"], vec![false, false], false)?;
            let out = out.column("cat")?;
            let cat = out.categorical()?;
            assert_order(cat, &["a", "a", "b", "c"]);

            let out = df.sort(["vals", "cat"], vec![false, false], false)?;
            let out = out.column("cat")?;
            let cat = out.categorical()?;
            assert_order(cat, &["b", "c", "a", "a"]);
        }
        Ok(())
    }
}
