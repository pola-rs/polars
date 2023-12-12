use std::borrow::Cow;

use super::*;
use crate::series::coalesce_series;

pub fn _join_suffix_name(name: &str, suffix: &str) -> String {
    format!("{name}{suffix}")
}

fn get_suffix(suffix: Option<&str>) -> &str {
    suffix.unwrap_or("_right")
}

/// Utility method to finish a join.
#[doc(hidden)]
pub fn _finish_join(
    mut df_left: DataFrame,
    mut df_right: DataFrame,
    suffix: Option<&str>,
) -> PolarsResult<DataFrame> {
    let mut left_names = PlHashSet::with_capacity(df_left.width());

    df_left.get_columns().iter().for_each(|series| {
        left_names.insert(series.name());
    });

    let mut rename_strs = Vec::with_capacity(df_right.width());

    df_right.get_columns().iter().for_each(|series| {
        if left_names.contains(series.name()) {
            rename_strs.push(series.name().to_owned())
        }
    });
    let suffix = get_suffix(suffix);

    for name in rename_strs {
        df_right.rename(&name, &_join_suffix_name(&name, suffix))?;
    }

    drop(left_names);
    df_left.hstack_mut(df_right.get_columns())?;
    Ok(df_left)
}

pub(super) fn coalesce_outer_join(
    mut df: DataFrame,
    keys_left: &[&str],
    keys_right: &[&str],
    suffix: Option<&str>,
) -> DataFrame {
    let schema = df.schema();
    let mut to_remove = Vec::with_capacity(keys_right.len());

    // SAFETY: we maintain invariants.
    let columns = unsafe { df.get_columns_mut() };
    for (&l, &r) in keys_left.iter().zip(keys_right.iter()) {
        let pos_l = schema.get_full(l).unwrap().0;

        let r = if l == r {
            let suffix = get_suffix(suffix);
            Cow::Owned(_join_suffix_name(r, suffix))
        } else {
            Cow::Borrowed(r)
        };
        let pos_r = schema.get_full(&r).unwrap().0;

        let l = columns[pos_l].clone();
        let r = columns[pos_r].clone();

        columns[pos_l] = coalesce_series(&[l, r]).unwrap();
        to_remove.push(pos_r);
    }
    // sort in reverse order, so the indexes remain correct if we remove.
    to_remove.sort_by(|a, b| b.cmp(a));
    for pos in to_remove {
        let _ = columns.remove(pos);
    }
    df
}

#[cfg(feature = "chunked_ids")]
pub(crate) fn create_chunked_index_mapping(chunks: &[ArrayRef], len: usize) -> Vec<ChunkId> {
    let mut vals = Vec::with_capacity(len);

    for (chunk_i, chunk) in chunks.iter().enumerate() {
        vals.extend((0..chunk.len()).map(|array_i| [chunk_i as IdxSize, array_i as IdxSize]))
    }

    vals
}
