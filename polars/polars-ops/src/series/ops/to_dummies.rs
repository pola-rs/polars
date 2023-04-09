use polars_core::frame::groupby::GroupsIndicator;

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
        values: Option<&Vec<AnyValue>>,
        unknown_value_identifier: Option<&str>,
    ) -> PolarsResult<DataFrame>;
}

impl ToDummies for Series {
    fn to_dummies(
        &self,
        separator: Option<&str>,
        values: Option<&Vec<AnyValue>>,
        unknown_value_identifier: Option<&str>,
    ) -> PolarsResult<DataFrame> {
        let sep = separator.unwrap_or("_");
        let col_name = self.name();
        let len = self.len();
        let groups = self.group_tuples(true, false)?;

        // Safety: groups are known to be in bounds
        let seen_values = unsafe { self.agg_first(&groups) };
        let groups_iter = || seen_values.iter().zip(groups.iter());

        let columns = match values {
            Some(values) => {
                // If there are predefined values, building the output columns is a little more
                // involved.
                let known_values = values
                    .iter()
                    .map(|av| av.to_owned())
                    .collect::<PlHashSet<_>>();
                let seen_values_set: PlHashSet<_> = seen_values.iter().collect();
                let unseen_values = known_values.difference(&seen_values_set);

                // First, if we are interested in unknown values (i.e. an
                // `unknown_value_identifier` is set), we create an additional column with the
                // union of the dummy identifiers for all unknown values.
                let unknown_column = match unknown_value_identifier {
                    Some(identifier) => {
                        let unknown_iter = groups_iter()
                            .filter(|(av, _)| {
                                // Do not consider null value as "unknown"
                                !known_values.contains(av) && !matches!(av, AnyValue::Null)
                            })
                            .map(|(_, group)| group);
                        Some(dummy_series_with_name(
                            &format!("{col_name}{sep}{identifier}"),
                            self.len(),
                            unknown_iter,
                        ))
                    }
                    None => None,
                };

                // In order to add columns for seen and unseen values, we proceed in two steps:
                // 1: remove unknown values
                let seen_iter = groups_iter().filter(|(av, _)| known_values.contains(av));

                // 2: add groups which have not been encountered. Here, we use empty index groups
                // to represent "no indicators"
                let empty = vec![];
                let unseen_iter = unseen_values
                    .into_iter()
                    .map(|av| ((*av).clone(), GroupsIndicator::Idx((0, &empty))));

                let columns =
                    dummy_series_for_values(col_name, len, seen_iter.chain(unseen_iter), sep);

                // Eventually, we add the column with the "unknown" indicators if available
                if let Some(column) = unknown_column {
                    columns.chain(std::iter::once(column)).collect()
                } else {
                    columns.collect()
                }
            }
            None => {
                // If there are no predefined values, we can simply build the series from the
                // groups we found.
                dummy_series_for_values(col_name, len, groups_iter(), sep).collect()
            }
        };

        Ok(DataFrame::new_no_checks(sort_columns(columns)))
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

fn dummy_series_for_values<'a>(
    col_name: &'a str,
    len: usize,
    values_and_groups: impl Iterator<Item = (AnyValue<'a>, GroupsIndicator<'a>)> + 'a,
    sep: &'a str,
) -> impl Iterator<Item = Series> + 'a {
    values_and_groups.map(move |(av, group)| {
        let name = format_value(&av, col_name, sep);
        dummy_series_with_name(&name, len, std::iter::once(group))
    })
}

#[inline]
fn dummy_series_with_name<'a>(
    name: &str,
    len: usize,
    groups: impl Iterator<Item = GroupsIndicator<'a>>,
) -> Series {
    let mut av = vec![0 as DummyType; len];
    for group in groups {
        match group {
            GroupsIndicator::Idx((_, indices)) => {
                dummies_idx_insert(&mut av, indices);
            }
            GroupsIndicator::Slice([offset, len]) => {
                dummies_slice_insert(&mut av, offset, len);
            }
        };
    }
    DummyCa::from_vec(name, av).into_series()
}

#[inline]
fn dummies_idx_insert(av: &mut [DummyType], indices: &[IdxSize]) {
    for &idx in indices {
        let elem = unsafe { av.get_unchecked_mut(idx as usize) };
        *elem = 1;
    }
}

#[inline]
fn dummies_slice_insert(av: &mut [DummyType], group_offset: IdxSize, group_len: IdxSize) {
    for idx in group_offset..(group_offset + group_len) {
        let elem = unsafe { av.get_unchecked_mut(idx as usize) };
        *elem = 1;
    }
}

fn sort_columns(mut columns: Vec<Series>) -> Vec<Series> {
    columns.sort_by(|a, b| a.name().partial_cmp(b.name()).unwrap());
    columns
}
