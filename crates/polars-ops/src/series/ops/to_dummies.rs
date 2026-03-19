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
        drop_nulls: bool,
    ) -> PolarsResult<DataFrame>;
}

impl ToDummies for Series {
    fn to_dummies(
        &self,
        separator: Option<&str>,
        drop_first: bool,
        drop_nulls: bool,
    ) -> PolarsResult<DataFrame> {
        let sep = separator.unwrap_or("_");
        let col_name = self.name();

        // We only need to maintain order if we need to drop the first non-null item.
        let maintain_order = drop_first;
        let groups = self.group_tuples(true, maintain_order)?;

        // SAFETY: groups are in bounds.
        let columns = unsafe { self.agg_first(&groups) };
        let columns = columns.iter().zip(groups.iter());
        let mut seen_first = false;
        let columns = columns
            .filter_map(|(av, group)| {
                if av.is_null() && drop_nulls {
                    return None;
                } else if !seen_first && !av.is_null() && drop_first {
                    // The position of the first non-null item could be either 0 or 1.
                    seen_first = true;
                    return None;
                }
                // strings are formatted with extra \" \" in polars, so we
                // extract the string
                let name = if let Some(s) = av.get_str() {
                    format_pl_smallstr!("{col_name}{sep}{s}")
                } else {
                    // other types don't have this formatting issue
                    format_pl_smallstr!("{col_name}{sep}{av}")
                };

                let ca = match group {
                    GroupsIndicator::Idx((_, group)) => dummies_helper_idx(group, self.len(), name),
                    GroupsIndicator::Slice([offset, len]) => {
                        dummies_helper_slice(offset, len, self.len(), name)
                    },
                };
                Some(ca.into_column())
            })
            .collect::<Vec<_>>();

        DataFrame::new_infer_height(sort_columns(columns))
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
