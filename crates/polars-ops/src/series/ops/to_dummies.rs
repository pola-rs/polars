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
    fn to_dummies(&self, separator: Option<&str>, drop_first: bool) -> PolarsResult<DataFrame>;
}

impl ToDummies for Series {
    fn to_dummies(&self, separator: Option<&str>, drop_first: bool) -> PolarsResult<DataFrame> {
        let sep = separator.unwrap_or("_");
        let col_name = self.name();
        let groups = self.group_tuples(true, drop_first)?;

        // SAFETY: groups are in bounds
        let columns = unsafe { self.agg_first(&groups) };
        let columns = columns.iter().zip(groups.iter()).skip(drop_first as usize);
        let columns = columns
            .map(|(av, group)| {
                // strings are formatted with extra \" \" in polars, so we
                // extract the string
                let name = if let Some(s) = av.get_str() {
                    format!("{col_name}{sep}{s}")
                } else {
                    // other types don't have this formatting issue
                    format!("{col_name}{sep}{av}")
                };

                let ca = match group {
                    GroupsIndicator::Idx((_, group)) => {
                        dummies_helper_idx(group, self.len(), &name)
                    },
                    GroupsIndicator::Slice([offset, len]) => {
                        dummies_helper_slice(offset, len, self.len(), &name)
                    },
                };
                ca.into_series()
            })
            .collect();

        Ok(unsafe { DataFrame::new_no_checks(sort_columns(columns)) })
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

fn sort_columns(mut columns: Vec<Series>) -> Vec<Series> {
    columns.sort_by(|a, b| a.name().partial_cmp(b.name()).unwrap());
    columns
}
