use indexmap::IndexMap;
use polars_utils::format_pl_smallstr;

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
        drop_first: bool,
        categories: Option<&Vec<PlSmallStr>>,
    ) -> PolarsResult<DataFrame>;
}

impl ToDummies for Series {
    fn to_dummies(
        &self,
        separator: Option<&str>,
        drop_first: bool,
        categories: Option<&Vec<PlSmallStr>>,
    ) -> PolarsResult<DataFrame> {
        let sep = separator.unwrap_or("_");
        let col_name = self.name();
        let groups = self.group_tuples(true, drop_first)?;

        // SAFETY: groups are in bounds
        let cols = unsafe { self.agg_first(&groups) };
        let cols = cols.iter().zip(groups.iter());

        let mut columns: IndexMap<String, GroupsIndicator<'_>> = IndexMap::new();
        for (av, group) in cols {
            let name = match av.get_str() {
                Some(s) => s.to_string(),
                None => format!("{av}"),
            };
            if columns.contains_key(&name) {
                polars_bail!(Duplicate: "column with name '{name}' has more than one occurrence")
            }
            columns.insert(name, group);
        }

        let columns = match categories {
            Some(cats) => {
                // if categories are provided, we create dummies for only those categories, but all
                // of them, even if they are not present in the data
                // The resulting columns stay in the same order as the categories
                cats.iter()
                    .skip(drop_first as usize)
                    .map(|cat| {
                        let name = format_pl_smallstr!("{col_name}{sep}{cat}");
                        match columns.get(cat.as_str()) {
                            // // category observed in data, default case
                            Some(GroupsIndicator::Idx((_, idxs))) => {
                                dummies_helper_idx(idxs, self.len(), name).into_column()
                            },
                            Some(GroupsIndicator::Slice([offset, len])) => {
                                dummies_helper_slice(*offset, *len, self.len(), name).into_column()
                            },
                            // category not present in data -> all-zero column
                            None => UInt8Chunked::full(name, 0u8, self.len()).into_column(),
                        }
                    })
                    .collect::<Vec<_>>()
            },
            None => sort_columns(
                columns
                    .iter()
                    .skip(drop_first as usize)
                    .map(|(name, group)| {
                        let name = format_pl_smallstr!("{col_name}{sep}{name}");

                        let ca = match group {
                            GroupsIndicator::Idx((_, group)) => {
                                dummies_helper_idx(group, self.len(), name)
                            },
                            GroupsIndicator::Slice([offset, len]) => {
                                dummies_helper_slice(*offset, *len, self.len(), name)
                            },
                        };
                        ca.into_column()
                    })
                    .collect::<Vec<_>>(),
            ),
        };

        DataFrame::new(columns)
    }
}

fn dummies_helper_idx(groups: &[IdxSize], len: usize, name: PlSmallStr) -> DummyCa {
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
    name: PlSmallStr,
) -> DummyCa {
    let mut av = vec![0 as DummyType; len];

    for idx in group_offset..(group_offset + group_len) {
        let elem = unsafe { av.get_unchecked_mut(idx as usize) };
        *elem = 1;
    }

    ChunkedArray::from_vec(name, av)
}

fn sort_columns(mut columns: Vec<Column>) -> Vec<Column> {
    columns.sort_by(|a, b| a.name().partial_cmp(b.name()).unwrap());
    columns
}
