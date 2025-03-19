use polars_utils::format_pl_smallstr;

use super::*;
use crate::series::coalesce_columns;

pub fn _join_suffix_name(name: &str, suffix: &str) -> PlSmallStr {
    format_pl_smallstr!("{name}{suffix}")
}

fn get_suffix(suffix: Option<PlSmallStr>) -> PlSmallStr {
    suffix.unwrap_or_else(|| PlSmallStr::from_static("_right"))
}

/// Renames the columns on the right to not clash with the left using a specified or otherwise default suffix
/// and then merges the right dataframe into the left
#[doc(hidden)]
pub fn _finish_join(
    mut df_left: DataFrame,
    mut df_right: DataFrame,
    suffix: Option<PlSmallStr>,
) -> PolarsResult<DataFrame> {
    let mut left_names = PlHashSet::with_capacity(df_left.width());

    df_left.get_columns().iter().for_each(|series| {
        left_names.insert(series.name());
    });

    let mut rename_strs = Vec::with_capacity(df_right.width());
    let right_names = df_right.schema();

    for name in right_names.iter_names() {
        if left_names.contains(name) {
            rename_strs.push(name.clone())
        }
    }

    let suffix = get_suffix(suffix);

    for name in rename_strs {
        let new_name = _join_suffix_name(name.as_str(), suffix.as_str());
        // Safety: IR resolving should guarantee this passes
        df_right.rename(&name, new_name.clone()).unwrap();
    }

    drop(left_names);
    // Safety: IR resolving should guarantee this passes
    unsafe { df_left.hstack_mut_unchecked(df_right.get_columns()) };
    Ok(df_left)
}

pub fn _coalesce_full_join(
    mut df: DataFrame,
    keys_left: &[PlSmallStr],
    keys_right: &[PlSmallStr],
    suffix: Option<PlSmallStr>,
    df_left: &DataFrame,
) -> DataFrame {
    // No need to allocate the schema because we already
    // know for certain that the column name for left is `name`
    // and for right is `name + suffix`
    let schema_left = if keys_left == keys_right {
        Arc::new(Schema::default())
    } else {
        df_left.schema().clone()
    };

    let schema = df.schema().clone();
    let mut to_remove = Vec::with_capacity(keys_right.len());

    // SAFETY: we maintain invariants.
    let columns = unsafe { df.get_columns_mut() };
    let suffix = get_suffix(suffix);
    for (l, r) in keys_left.iter().zip(keys_right.iter()) {
        let pos_l = schema.get_full(l.as_str()).unwrap().0;

        let r = if l == r || schema_left.contains(r.as_str()) {
            _join_suffix_name(r.as_str(), suffix.as_str())
        } else {
            r.clone()
        };
        let pos_r = schema.get_full(&r).unwrap().0;

        let l = columns[pos_l].clone();
        let r = columns[pos_r].clone();

        columns[pos_l] = coalesce_columns(&[l, r]).unwrap();
        to_remove.push(pos_r);
    }
    // sort in reverse order, so the indexes remain correct if we remove.
    to_remove.sort_by(|a, b| b.cmp(a));
    for pos in to_remove {
        let _ = columns.remove(pos);
    }
    df.clear_schema();
    df
}

#[cfg(feature = "chunked_ids")]
pub(crate) fn create_chunked_index_mapping(chunks: &[ArrayRef], len: usize) -> Vec<ChunkId> {
    let mut vals = Vec::with_capacity(len);

    for (chunk_i, chunk) in chunks.iter().enumerate() {
        vals.extend(
            (0..chunk.len()).map(|array_i| ChunkId::store(chunk_i as IdxSize, array_i as IdxSize)),
        )
    }

    vals
}
