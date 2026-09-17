use polars_arrow::bitmap::Bitmap;
use polars_core::prelude::*;
use polars_utils::IdxSize;

/// Look up `key` in every row of `ca`, returning the Map's value dtype.
///
/// Missing keys, null keys and null Map rows yield null.
pub fn map_get(ca: &MapChunked, key: &Series) -> PolarsResult<Series> {
    with_broadcast_map(ca, key, |ca, key| {
        let idx = map_key_index(ca, key)?;
        // SAFETY: indices address the same live entries flattened by `values`.
        Ok(unsafe { ca.values().take_unchecked(&idx) })
    })
}

/// Check each row of `ca` for `key`.
///
/// Null Map rows stay null; null keys never match.
pub fn map_contains_key(ca: &MapChunked, key: &Series) -> PolarsResult<BooleanChunked> {
    with_broadcast_map(ca, key, |ca, key| {
        let found = map_key_index(ca, key)?.is_not_null();
        // Restore null rows, which `is_not_null` turned into false.
        Ok(found.with_validity(ca.storage().rechunk_validity()))
    })
}

/// Support a length-1 Map or key input.
///
/// Only the Map is expanded here; [`map_key_index`] handles scalar keys.
fn with_broadcast_map<T>(
    ca: &MapChunked,
    key: &Series,
    f: impl FnOnce(&MapChunked, &Series) -> PolarsResult<T>,
) -> PolarsResult<T> {
    match (ca.len(), key.len()) {
        (map_len, key_len) if map_len == key_len => f(ca, key),
        (_, 1) => f(ca, key),
        (1, key_len) => {
            let broadcast = ca.clone().into_series().new_from_index(0, key_len);
            f(broadcast.map()?, key)
        },
        (map_len, key_len) => polars_bail!(
            ShapeMismatch:
            "cannot look up {key_len} keys in {map_len} maps: lengths must match, \
            unless either side is a single value",
        ),
    }
}

/// Find each key's flat entry index; return null for missing keys or null rows.
///
/// [`MapChunked::keys`] and [`MapChunked::values`] share the same live entry order.
/// `key` must have length 1 or one element per row.
///
/// Scans entries linearly; Map storage has no key index.
fn map_key_index(ca: &MapChunked, key: &Series) -> PolarsResult<IdxCa> {
    // Read the row lengths off the offsets rather than through `live_storage` or `key_lists`,
    // neither of which can hand them over without materializing something.
    let row_lengths = || ca.live_row_lengths();

    let flat_keys = ca.keys();
    if flat_keys.is_empty() {
        return Ok(IdxCa::full_null(ca.name().clone(), ca.len()));
    }

    // Repeat each row's key over its entries for a single flat comparison.
    let mask = if key.len() == 1 {
        flat_keys.equal_missing(key)?
    } else {
        let rows = IdxCa::from_vec(
            PlSmallStr::EMPTY,
            row_lengths()
                .enumerate()
                .flat_map(|(row, len)| std::iter::repeat_n(row as IdxSize, len))
                .collect(),
        );
        // SAFETY: the row indices come from the offsets of a Map of `key`'s length.
        flat_keys.equal_missing(&unsafe { key.take_unchecked(&rows) })?
    };
    // Map keys are non-null, so null query keys never match.
    let mask = mask.rechunk();
    let mask = mask.downcast_as_array();
    debug_assert!(
        mask.validity().is_none_or(|v| v.unset_bits() == 0),
        "`equal_missing` cannot yield nulls"
    );
    let mask: &Bitmap = mask.values();

    let mut start = 0;
    let index = row_lengths().map(|len| {
        let first_match = mask.clone().sliced(start, len).iter().take_leading_zeros();
        let index = (first_match < len).then(|| (start + first_match) as IdxSize);
        start += len;
        index
    });
    Ok(IdxCa::from_iter_options(ca.name().clone(), index))
}
