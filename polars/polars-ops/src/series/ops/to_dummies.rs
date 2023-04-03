use polars_core::frame::groupby::GroupsIndicator;
use polars_core::utils::rayon::prelude::ParallelIterator;

use super::*;

#[cfg(feature = "dtype-u8")]
type DummyType = u8;
#[cfg(feature = "dtype-u8")]
type DummyCa = UInt8Chunked;

#[cfg(not(feature = "dtype-u8"))]
type DummyType = i32;
#[cfg(not(feature = "dtype-u8"))]
type DummyCa = Int32Chunked;

pub trait ToDummies {
    fn to_dummies(
        &self,
        separator: Option<&str>,
        include_null: bool,
        values: Option<&Vec<AnyValue>>,
    ) -> PolarsResult<DataFrame>;
}

impl ToDummies for Series {
    fn to_dummies(
        &self,
        separator: Option<&str>,
        include_null: bool,
        values: Option<&Vec<AnyValue>>,
    ) -> PolarsResult<DataFrame> {
        let sep = separator.unwrap_or("_");
        let col_name = self.name();
        let groups = self.group_tuples(true, false)?;
        let valid_values = values.map(|v| v.into_iter().collect::<PlHashSet<_>>());

        // safety: groups are in bounds
        let values = unsafe { self.agg_first(&groups) };
        let mut columns: Vec<_> = values
            .iter()
            .zip(groups.iter())
            // This filter must be kept after the `.zip` to align iterators.
            .filter(|(av, _)| filter_value(av, include_null, &valid_values))
            .map(|(av, group)| {
                let name = format_value(&av, col_name, sep);
                let ca = match group {
                    GroupsIndicator::Idx((_, group)) => {
                        dummies_helper_idx(group, self.len(), &name)
                    }
                    GroupsIndicator::Slice([offset, len]) => {
                        dummies_helper_slice(offset, len, self.len(), &name)
                    }
                };
                ca.into_series()
            })
            .collect();

        // If we want to have a null value indicator column and null is not included in the
        // values, we need to add the column retrospectively.
        let null_value_exists = values.iter().any(|av| matches!(av, AnyValue::Null));
        if include_null && !null_value_exists {
            let name = format_value(&AnyValue::Null, col_name, sep);
            columns.push(dummies_empty(self.len(), &name));
        }

        // If the set of values has been predefined, we need to add empty columns for those that we
        // did not encounter in the data.
        if let Some(values_set) = valid_values {
            let own = values.iter().collect::<Vec<_>>();
            let seen_set = own.iter().collect();
            let new_columns: Vec<_> = values_set
                .par_difference(&seen_set)
                .map(|av| {
                    let name = format_value(av, col_name, sep);
                    dummies_empty(self.len(), &name)
                })
                .collect();
            columns.extend(new_columns);
        }

        Ok(DataFrame::new_no_checks(sort_columns(columns)))
    }
}

#[inline]
fn filter_value(
    av: &AnyValue,
    include_null: bool,
    valid_values: &Option<PlHashSet<&AnyValue>>,
) -> bool {
    match av {
        // We need to filter out `null` groups here if we don't want them to end up in the result.
        // TODO: move this filter to the groupby call once we can ignore NULL values there.
        // See https://github.com/pola-rs/polars/issues/7943 for progress.
        AnyValue::Null => include_null,
        // We also need to check whether the value is in the set of allowed values
        _ => match valid_values {
            None => true,
            Some(values_set) => values_set.contains(av),
        },
    }
}

#[inline]
fn format_value(av: &AnyValue, col_name: &str, sep: &str) -> String {
    // strings are formatted with extra \" \" in polars, so we
    // extract the string
    if let Some(s) = av.get_str() {
        format!("{col_name}{sep}{s}")
    } else {
        // other types don't have this formatting issue
        format!("{col_name}{sep}{av}")
    }
}

fn dummies_helper_idx(groups: &[IdxSize], len: usize, name: &str) -> DummyCa {
    let mut av = vec![0 as DummyType; len];

    for &idx in groups {
        let elem = unsafe { av.get_unchecked_mut(idx as usize) };
        *elem = 1;
    }

    ChunkedArray::from_vec(name, av)
}

fn dummies_helper_slice(
    group_offset: IdxSize,
    group_len: IdxSize,
    len: usize,
    name: &str,
) -> DummyCa {
    let mut av = vec![0 as DummyType; len];

    for idx in group_offset..(group_offset + group_len) {
        let elem = unsafe { av.get_unchecked_mut(idx as usize) };
        *elem = 1;
    }

    ChunkedArray::from_vec(name, av)
}

fn dummies_empty(len: usize, name: &str) -> Series {
    DummyCa::from_vec(name, vec![0 as DummyType; len]).into_series()
}

fn sort_columns(mut columns: Vec<Series>) -> Vec<Series> {
    columns.sort_by(|a, b| a.name().partial_cmp(b.name()).unwrap());
    columns
}
