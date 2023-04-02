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
    fn to_dummies(&self, separator: Option<&str>, include_null: bool) -> PolarsResult<DataFrame>;
}

impl ToDummies for Series {
    fn to_dummies(&self, separator: Option<&str>, include_null: bool) -> PolarsResult<DataFrame> {
        let sep = separator.unwrap_or("_");
        let col_name = self.name();
        let groups = self.group_tuples(true, false)?;

        // safety: groups are in bounds
        let values = unsafe { self.agg_first(&groups) };
        let mut columns: Vec<_> = values
            .iter()
            .zip(groups.iter())
            // We need to filter out `null` groups here if we don't want them to end up in the
            // result. Take care to keep this filter after the `.zip` to align iterators.
            // TODO: move this filter to the groupby call once we can ignore NULL values there.
            // See https://github.com/pola-rs/polars/issues/7943 for progress.
            .filter(|(av, _)| include_null || !matches!(av, AnyValue::Null))
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
            columns.push(
                DummyCa::from_vec(
                    &format!("{col_name}{sep}null"),
                    vec![0 as DummyType; self.len()],
                )
                .into_series(),
            )
        }

        Ok(DataFrame::new_no_checks(sort_columns(columns)))
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
