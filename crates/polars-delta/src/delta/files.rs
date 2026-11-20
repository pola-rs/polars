//! Data files of a snapshot, kept in the representation they were read in (typed commit rows,
//! checkpoint columns) and reconciled by key, so that only the survivors are ever copied.
use polars_arrow::array::*;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::offset::OffsetsBuffer;
use polars_utils::aliases::PlHashSet;

use super::actions::{AddRec, DvId, DvRec, Key, RemoveRec};
use super::error::{Result, invalid};
use super::stats::{self, FileStats};

/// A deletion vector descriptor borrowed from its source.
#[derive(Debug, Clone, Copy)]
pub struct DvView<'a> {
    pub id: DvId<'a>,
    pub size_in_bytes: Option<u64>,
    pub cardinality: Option<u64>,
}
impl<'a> From<&'a DvRec> for DvView<'a> {
    fn from(dv: &'a DvRec) -> Self {
        Self {
            id: dv.id(),
            size_in_bytes: dv.size_in_bytes,
            cardinality: dv.cardinality,
        }
    }
}
pub struct Rows {
    pub adds: Vec<AddRec>,
    pub removes: Vec<RemoveRec>,
}
/// A run of rows of one checkpoint file: the `add` struct column of a row group.
pub struct CheckpointFiles {
    len: usize,
    add_valid: Option<Bitmap>,
    path: Box<dyn Array>,
    size: Box<dyn Array>,
    pv_offsets: OffsetsBuffer<i32>,
    pv_keys: Box<dyn Array>,
    pv_values: Box<dyn Array>,
    stats: Option<Box<dyn Array>>,
    stats_parsed: Option<StructArray>,
    dv: Option<DvColumns>,
}
struct DvColumns {
    validity: Option<Bitmap>,
    storage_type: Option<Box<dyn Array>>,
    path_or_inline: Option<Box<dyn Array>>,
    offset: Option<Box<dyn Array>>,
    size_in_bytes: Option<Box<dyn Array>>,
    cardinality: Option<Box<dyn Array>>,
}
pub enum Source {
    Rows(Rows),
    Checkpoint(Box<CheckpointFiles>),
}
fn child<'a>(s: &'a StructArray, name: &str) -> Option<&'a dyn Array> {
    s.fields()
        .iter()
        .position(|f| f.name == name)
        .map(|i| s.values()[i].as_ref())
}
fn child_box(s: &StructArray, name: &str) -> Option<Box<dyn Array>> {
    child(s, name).map(|a| a.to_boxed())
}
pub fn str_at(a: &dyn Array, i: usize) -> Option<&str> {
    if a.is_null(i) {
        return None;
    }
    match a.dtype() {
        ArrowDataType::Utf8View => Some(a.as_any().downcast_ref::<Utf8ViewArray>()?.value(i)),
        ArrowDataType::Utf8 => Some(a.as_any().downcast_ref::<Utf8Array<i32>>()?.value(i)),
        ArrowDataType::LargeUtf8 => Some(a.as_any().downcast_ref::<Utf8Array<i64>>()?.value(i)),
        _ => None,
    }
}
pub fn int_at(a: &dyn Array, i: usize) -> Option<i64> {
    if a.is_null(i) {
        return None;
    }
    macro_rules! at {
        ($t:ty) => {
            Some(i64::from(
                a.as_any().downcast_ref::<PrimitiveArray<$t>>()?.value(i),
            ))
        };
    }
    match a.dtype() {
        ArrowDataType::Int8 => at!(i8),
        ArrowDataType::Int16 => at!(i16),
        ArrowDataType::Int32 => at!(i32),
        ArrowDataType::Int64 => Some(a.as_any().downcast_ref::<PrimitiveArray<i64>>()?.value(i)),
        ArrowDataType::UInt8 => at!(u8),
        ArrowDataType::UInt16 => at!(u16),
        ArrowDataType::UInt32 => at!(u32),
        ArrowDataType::UInt64 => {
            i64::try_from(a.as_any().downcast_ref::<PrimitiveArray<u64>>()?.value(i)).ok()
        },
        _ => None,
    }
}
impl CheckpointFiles {
    /// `add` is the (projected) `add` column of one row group. Rows where it is null are not adds.
    pub fn new(add: &StructArray) -> Result<Self> {
        let add_valid = add.validity().cloned();
        let len = add.len();
        let is_add = |i: usize| add_valid.as_ref().is_none_or(|v| v.get_bit(i));
        let missing = |what: &str| invalid(format!("checkpoint add action without {what}"));
        let any_add = (0..len).any(is_add);
        let path = child_box(add, "path");
        let size = child_box(add, "size");
        let pv = child(add, "partitionValues");
        if any_add && (path.is_none() || size.is_none() || pv.is_none()) {
            return Err(missing("path, size or partitionValues"));
        }
        let (path, size) = match (path, size) {
            (Some(p), Some(s)) => (p, s),
            _ => (
                new_empty_array(ArrowDataType::Utf8View),
                new_empty_array(ArrowDataType::Int64),
            ),
        };
        let (pv_offsets, pv_keys, pv_values) = match pv {
            Some(pv) => {
                let map = pv
                    .as_any()
                    .downcast_ref::<MapArray>()
                    .ok_or_else(|| invalid("checkpoint partitionValues is not a map"))?;
                let entries = map
                    .field()
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| invalid("checkpoint partitionValues entries"))?;
                let [keys, values] = entries.values() else {
                    return Err(invalid("checkpoint partitionValues entries"));
                };
                (map.offsets().clone(), keys.to_boxed(), values.to_boxed())
            },
            None => (
                OffsetsBuffer::new(),
                new_empty_array(ArrowDataType::Utf8View),
                new_empty_array(ArrowDataType::Utf8View),
            ),
        };
        for i in (0..len).filter(|i| is_add(*i)) {
            if str_at(path.as_ref(), i).is_none() {
                return Err(invalid("missing/invalid path"));
            }
            if int_at(size.as_ref(), i).is_none_or(|s| s < 0) {
                return Err(invalid("missing/invalid size"));
            }
            if pv.is_some_and(|pv| pv.is_null(i)) {
                return Err(invalid("missing add.partitionValues"));
            }
        }
        let dv = child(add, "deletionVector")
            .map(|dv| {
                let dv = dv
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| invalid("checkpoint deletionVector is not a struct"))?;
                Ok::<_, polars_io_ext_ffi::common::FfiError>(DvColumns {
                    validity: dv.validity().cloned(),
                    storage_type: child_box(dv, "storageType"),
                    path_or_inline: child_box(dv, "pathOrInlineDv"),
                    offset: child_box(dv, "offset"),
                    size_in_bytes: child_box(dv, "sizeInBytes"),
                    cardinality: child_box(dv, "cardinality"),
                })
            })
            .transpose()?;
        let stats_parsed = child(add, "stats_parsed")
            .map(|s| {
                s.as_any()
                    .downcast_ref::<StructArray>()
                    .cloned()
                    .ok_or_else(|| invalid("checkpoint stats_parsed is not a struct"))
            })
            .transpose()?;
        Ok(Self {
            len,
            add_valid,
            path,
            size,
            pv_offsets,
            pv_keys,
            pv_values,
            stats: child_box(add, "stats"),
            stats_parsed,
            dv,
        })
    }
    fn is_add(&self, row: usize) -> bool {
        self.add_valid.as_ref().is_none_or(|v| v.get_bit(row))
    }
    fn dv(&self, row: usize) -> Result<Option<DvView<'_>>> {
        let Some(dv) = &self.dv else {
            return Ok(None);
        };
        if dv.validity.as_ref().is_some_and(|v| !v.get_bit(row)) {
            return Ok(None);
        }
        fn text<'a>(c: &'a Option<Box<dyn Array>>, row: usize, what: &str) -> Result<&'a str> {
            c.as_deref()
                .and_then(|c| str_at(c, row))
                .ok_or_else(|| invalid(format!("missing/invalid {what}")))
        }
        let int = |c: &Option<Box<dyn Array>>| c.as_deref().and_then(|c| int_at(c, row));
        let to_u64 = |v: Option<i64>, what: &str| {
            v.map(|v| u64::try_from(v).map_err(|_| invalid(format!("negative/invalid {what}"))))
                .transpose()
        };
        Ok(Some(DvView {
            id: DvId {
                storage_type: text(&dv.storage_type, row, "storageType")?,
                path_or_inline: text(&dv.path_or_inline, row, "pathOrInlineDv")?,
                offset: to_u64(int(&dv.offset), "DV offset")?,
            },
            size_in_bytes: to_u64(int(&dv.size_in_bytes), "sizeInBytes")?,
            cardinality: to_u64(int(&dv.cardinality), "cardinality")?,
        }))
    }
}
/// Where the requested columns' bounds are in a checkpoint's `stats_parsed`.
pub struct ParsedLookup<'a> {
    num_records: Option<&'a dyn Array>,
    tight_bounds: Option<&'a dyn Array>,
    min: Vec<Option<&'a dyn Array>>,
    max: Vec<Option<&'a dyn Array>>,
    null_count: Vec<Option<&'a dyn Array>>,
}
impl<'a> ParsedLookup<'a> {
    fn new(parsed: &'a StructArray, wanted: &[&str]) -> Self {
        let columns = |kind: &str| {
            let kind = child(parsed, kind).and_then(|k| k.as_any().downcast_ref::<StructArray>());
            wanted
                .iter()
                .map(|w| kind.and_then(|k| child(k, w)))
                .collect()
        };
        Self {
            num_records: child(parsed, "numRecords"),
            tight_bounds: child(parsed, "tightBounds"),
            min: columns("minValues"),
            max: columns("maxValues"),
            null_count: columns("nullCount"),
        }
    }
}
/// Per-batch precomputation for [`Source::file_stats`].
pub struct StatsLookup<'a> {
    parsed: Option<ParsedLookup<'a>>,
}
impl Source {
    pub fn path(&self, row: usize) -> &str {
        match self {
            Source::Rows(r) => &r.adds[row].path,
            // Validated when the batch was made.
            Source::Checkpoint(c) => str_at(c.path.as_ref(), row).unwrap_or(""),
        }
    }
    pub fn size(&self, row: usize) -> u64 {
        match self {
            Source::Rows(r) => r.adds[row].size,
            Source::Checkpoint(c) => int_at(c.size.as_ref(), row).unwrap_or(0) as u64,
        }
    }
    /// The partition value of `physical`: `None` when the file has no entry for it.
    pub fn partition(&self, row: usize, physical: &str) -> Option<Option<&str>> {
        match self {
            Source::Rows(r) => r.adds[row]
                .partition_values
                .iter()
                .rev()
                .find(|(k, _)| &**k == physical)
                .map(|(_, v)| v.as_deref()),
            Source::Checkpoint(c) => {
                let (start, end) = c.pv_offsets.start_end(row);
                (start..end)
                    .rev()
                    .find(|j| str_at(c.pv_keys.as_ref(), *j) == Some(physical))
                    .map(|j| str_at(c.pv_values.as_ref(), j))
            },
        }
    }
    pub fn dv(&self, row: usize) -> Result<Option<DvView<'_>>> {
        match self {
            Source::Rows(r) => Ok(r.adds[row].dv.as_ref().map(DvView::from)),
            Source::Checkpoint(c) => c.dv(row),
        }
    }
    fn key(&self, row: usize) -> Result<Key<'_>> {
        Ok(Key {
            path: self.path(row),
            dv: self.dv(row)?.map(|d| d.id),
        })
    }
    pub fn stats_lookup<'a>(&'a self, wanted: &[&str]) -> StatsLookup<'a> {
        StatsLookup {
            parsed: match self {
                Source::Checkpoint(c) => c
                    .stats_parsed
                    .as_ref()
                    .map(|p| ParsedLookup::new(p, wanted)),
                _ => None,
            },
        }
    }
    /// The file's statistics for `wanted` (physical column names): from the typed `stats_parsed`
    /// when the checkpoint has it for this file, else from the `stats` text.
    pub fn file_stats<'a>(
        &'a self,
        row: usize,
        wanted: &[&str],
        lookup: &StatsLookup<'a>,
    ) -> Option<FileStats<'a>> {
        match self {
            Source::Rows(r) => stats::scan(r.adds[row].stats.as_deref()?, wanted),
            Source::Checkpoint(c) => {
                if let (Some(parsed), Some(lk)) = (&c.stats_parsed, &lookup.parsed) {
                    if !parsed.is_null(row) {
                        let mut out = FileStats::new(wanted.len());
                        out.num_records = lk
                            .num_records
                            .and_then(|a| int_at(a, row))
                            .and_then(|v| u64::try_from(v).ok());
                        out.tight_bounds = lk.tight_bounds.and_then(|a| {
                            a.as_any()
                                .downcast_ref::<BooleanArray>()
                                .filter(|b| !b.is_null(row))
                                .map(|b| b.value(row))
                        });
                        for (dst, src) in [
                            (&mut out.min, &lk.min),
                            (&mut out.max, &lk.max),
                            (&mut out.null_count, &lk.null_count),
                        ] {
                            for (d, s) in dst.iter_mut().zip(src) {
                                // Bounds of a type that holds none are unknown.
                                *d = s
                                    .and_then(|a| stats::array_value(a, row))
                                    .unwrap_or(super::values::Sv::Null);
                            }
                        }
                        return Some(out);
                    }
                }
                stats::scan(str_at(c.stats.as_deref()?, row)?, wanted)
            },
        }
    }
}
/// Reconcile file actions: `sources` newest first, the checkpoint's last. Returns the surviving
/// adds as `(source, row)`, sorted by path.
///
/// An add survives unless a newer action holds its logical key (path and deletion vector) or a
/// newer add holds its path. Removes of a checkpoint are tombstones for files the snapshot
/// never reads, so they are not part of the sources.
pub fn reconcile(sources: &[Source]) -> Result<Vec<(u32, u32)>> {
    let mut seen: PlHashSet<Key<'_>> = PlHashSet::default();
    let mut seen_paths: PlHashSet<&str> = PlHashSet::default();
    let mut checkpoint_keys: PlHashSet<Key<'_>> = PlHashSet::default();
    let mut survivors = Vec::new();
    for (s, source) in sources.iter().enumerate() {
        match source {
            Source::Rows(rows) => {
                for r in &rows.removes {
                    seen.insert(Key {
                        path: &r.path,
                        dv: r.dv.as_ref().map(DvRec::id),
                    });
                }
                for (row, a) in rows.adds.iter().enumerate() {
                    let key = Key {
                        path: &a.path,
                        dv: a.dv.as_ref().map(DvRec::id),
                    };
                    // Mark every newest add even when a later remove already suppressed it.
                    let newest = seen_paths.insert(&a.path);
                    if seen.insert(key) && newest {
                        survivors.push((s as u32, row as u32));
                    }
                }
            },
            Source::Checkpoint(c) => {
                for row in (0..c.len).filter(|r| c.is_add(*r)) {
                    let key = source.key(row)?;
                    if !checkpoint_keys.insert(key) {
                        return Err(invalid("duplicate checkpoint logical file"));
                    }
                    let newest = seen_paths.insert(key.path);
                    if newest && !seen.contains(&key) {
                        survivors.push((s as u32, row as u32));
                    }
                }
            },
        }
    }
    survivors.sort_unstable_by(|a, b| {
        sources[a.0 as usize]
            .path(a.1 as usize)
            .cmp(sources[b.0 as usize].path(b.1 as usize))
    });
    Ok(survivors)
}
/// Whether any row of the batch is an add (for checks on embedded file actions).
pub fn has_adds(source: &Source) -> bool {
    match source {
        Source::Rows(r) => !r.adds.is_empty(),
        Source::Checkpoint(c) => (0..c.len).any(|r| c.is_add(r)),
    }
}
