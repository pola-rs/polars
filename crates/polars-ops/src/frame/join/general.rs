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
        let new_name = _join_suffix_name(&name, suffix);
        df_right.rename(&name, new_name.as_str()).map_err(|_| {
            polars_err!(Duplicate: "column with name '{}' already exists\n\n\
            You may want to try:\n\
            - renaming the column prior to joining\n\
            - using the `suffix` parameter to specify a suffix different to the default one ('_right')", new_name)
        })?;
    }

    drop(left_names);
    df_left.hstack_mut(df_right.get_columns())?;
    Ok(df_left)
}

pub fn _coalesce_full_join(
    mut df: DataFrame,
    keys_left: &[&str],
    keys_right: &[&str],
    suffix: Option<&str>,
    df_left: &DataFrame,
) -> DataFrame {
    // No need to allocate the schema because we already
    // know for certain that the column name for left left is `name`
    // and for right is `name + suffix`
    let schema_left = if keys_left == keys_right {
        Schema::default()
    } else {
        df_left.schema()
    };

    let schema = df.schema();
    let mut to_remove = Vec::with_capacity(keys_right.len());

    // SAFETY: we maintain invariants.
    let columns = unsafe { df.get_columns_mut() };
    for (&l, &r) in keys_left.iter().zip(keys_right.iter()) {
        let pos_l = schema.get_full(l).unwrap().0;

        let r = if l == r || schema_left.contains(r) {
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
        vals.extend(
            (0..chunk.len()).map(|array_i| ChunkId::store(chunk_i as IdxSize, array_i as IdxSize)),
        )
    }

    vals
}
