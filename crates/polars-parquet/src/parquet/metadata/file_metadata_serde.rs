//! Custom `Serialize` / `Deserialize` for [`FileMetadata`].
//
// Wire form for IR-plan-borne metadata. Workers call
// `bincode::deserialize::<FileMetadata>(...)` and get a fully-formed
// (pruned) `FileMetadata` back; existing reader code runs unchanged.
//
// The wire DTO is a private, flat mirror of `Compact*` minus the stats
// `ByteRange`s (replaced with owned bytes). On deserialize the stat bytes
// are laid out into a fresh `footer_buf`, with new `ByteRange`s pointing
// into it; the existing `chunk.statistics(footer_buf)` accessor works
// unchanged.

use polars_buffer::Buffer;
use serde::{Deserialize, Serialize};

use super::compact::{
    ByteRange, CompactColumnChunk, CompactColumnMetaData, CompactRowGroup, CompactStatistics,
};
use super::schema_descriptor::SchemaDescriptor;
use super::{ColumnChunkMetadata, FileMetadata, RowGroupMetadata};
use crate::parquet::compression::Compression;

#[derive(Serialize, Deserialize)]
struct FileMetadataWire {
    version: i32,
    schema_descr: SchemaDescriptor,
    row_groups: Vec<RowGroupWire>,
}

#[derive(Serialize, Deserialize)]
struct RowGroupWire {
    num_rows: i64,
    sorting_columns: Option<Vec<SortingColumnWire>>,
    columns: Vec<ChunkWire>,
}

#[derive(Serialize, Deserialize, Clone)]
struct SortingColumnWire {
    column_idx: i32,
    descending: bool,
    nulls_first: bool,
}

impl From<&polars_parquet_format::SortingColumn> for SortingColumnWire {
    fn from(s: &polars_parquet_format::SortingColumn) -> Self {
        Self {
            column_idx: s.column_idx,
            descending: s.descending,
            nulls_first: s.nulls_first,
        }
    }
}

impl From<SortingColumnWire> for polars_parquet_format::SortingColumn {
    fn from(s: SortingColumnWire) -> Self {
        Self {
            column_idx: s.column_idx,
            descending: s.descending,
            nulls_first: s.nulls_first,
        }
    }
}

/// Wire entry per chunk. Drops fields with zero read-path consumers
/// (`total_uncompressed_size`, `index_page_offset`, bloom/page-index
/// pointers) and collapses (data_offset, dict_offset, compressed_size)
/// to (chunk_offset, chunk_size). On deserialize the reconstructed
/// `byte_range()` is bit-identical (data_page_offset = chunk_offset,
/// dict_offset = None, total_compressed_size = chunk_size).
#[derive(Serialize, Deserialize)]
struct ChunkWire {
    codec: Compression,
    num_values: i64,
    chunk_offset: i64,
    chunk_size: i64,
    statistics: Option<StatWire>,
}

/// Stat wire entry. Drops fields with zero read-path consumers:
/// - `distinct_count`: 0 callers in polars's read or predicate paths
/// - `is_min_value_exact` / `is_max_value_exact`: 0 callers
///
/// On deserialize these are set to `None`.
#[derive(Serialize, Deserialize)]
struct StatWire {
    null_count: Option<i64>,
    min_value: Option<Vec<u8>>,
    max_value: Option<Vec<u8>>,
}

impl Serialize for FileMetadata {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let footer = &self.footer_buf;

        let row_groups: Vec<RowGroupWire> = self
            .row_groups
            .iter()
            .map(|rg| rg_to_wire(rg, footer))
            .collect();

        let wire = FileMetadataWire {
            version: self.version,
            schema_descr: self.schema_descr.clone(),
            row_groups,
        };
        wire.serialize(s)
    }
}

fn rg_to_wire(rg: &RowGroupMetadata, footer: &[u8]) -> RowGroupWire {
    let columns = rg
        .parquet_columns()
        .iter()
        .map(|c| chunk_to_wire(c, footer))
        .collect();

    RowGroupWire {
        num_rows: rg.num_rows() as i64,
        sorting_columns: rg
            .sorting_columns()
            .map(|sc| sc.iter().map(SortingColumnWire::from).collect()),
        columns,
    }
}

fn chunk_to_wire(c: &ColumnChunkMetadata, footer: &[u8]) -> ChunkWire {
    let m = c.compact_metadata();
    let statistics = m.statistics.as_ref().map(|s| StatWire {
        null_count: s.null_count,
        min_value: s.min_value.map(|r| r.resolve(footer).to_vec()),
        max_value: s.max_value.map(|r| r.resolve(footer).to_vec()),
    });
    // Pre-resolve byte_range so wire form has just (offset, len). The
    // reader's only consumers of these fields go through `byte_range()`,
    // which already does this resolution.
    let byte_range = c.byte_range();
    ChunkWire {
        codec: m.codec,
        num_values: m.num_values,
        chunk_offset: byte_range.start as i64,
        chunk_size: m.total_compressed_size,
        statistics,
    }
}

impl<'de> Deserialize<'de> for FileMetadata {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let wire = FileMetadataWire::deserialize(d)?;

        // Rebuild a synthesised footer buffer. Lay out all stat bytes
        // back-to-back; CompactStatistics' `ByteRange`s point into it.
        // Total length is bounded by 2 × num_chunks × max_stat_blob_size,
        // which after pruning is small (KBs).
        let footer_len: usize = wire
            .row_groups
            .iter()
            .flat_map(|rg| rg.columns.iter())
            .filter_map(|c| c.statistics.as_ref())
            .map(|s| {
                s.min_value.as_ref().map_or(0, |v| v.len())
                    + s.max_value.as_ref().map_or(0, |v| v.len())
            })
            .sum();

        let mut footer = Vec::with_capacity(footer_len);

        // Build CompactRowGroups, threading the shared footer buffer.
        let row_groups_compact: Vec<CompactRowGroup> = wire
            .row_groups
            .into_iter()
            .map(|rg| rg_from_wire(rg, &mut footer))
            .collect();

        // Total num_rows
        let num_rows: i64 = row_groups_compact.iter().map(|rg| rg.num_rows).sum();

        // Wire form already carries a built SchemaDescriptor; per-row-group
        // metadata is rebuilt via RowGroupMetadata::from_compact below.
        let schema_descr = wire.schema_descr;

        let footer_buf = Buffer::from_vec(footer);

        let mut max_row_group_height = 0;
        let row_groups: Vec<RowGroupMetadata> = row_groups_compact
            .into_iter()
            .map(|rg| {
                let md = RowGroupMetadata::from_compact(&schema_descr, rg)
                    .map_err(serde::de::Error::custom)?;
                max_row_group_height = max_row_group_height.max(md.num_rows());
                Ok(md)
            })
            .collect::<Result<_, D::Error>>()?;

        Ok(FileMetadata {
            version: wire.version,
            num_rows: usize::try_from(num_rows)
                .map_err(|e| serde::de::Error::custom(format!("num_rows overflow: {e}")))?,
            max_row_group_height,
            created_by: None,
            row_groups,
            key_value_metadata: None,
            schema_descr,
            column_orders: None,
            footer_buf,
        })
    }
}

fn rg_from_wire(rg: RowGroupWire, footer: &mut Vec<u8>) -> CompactRowGroup {
    let columns = rg
        .columns
        .into_iter()
        .map(|c| chunk_from_wire(c, footer))
        .collect();
    CompactRowGroup {
        columns,
        // total_byte_size is dropped on the wire (0 read-path callers).
        // Set to 0; if any caller needs it later they can recompute from
        // chunk byte_ranges.
        total_byte_size: 0,
        num_rows: rg.num_rows,
        sorting_columns: rg
            .sorting_columns
            .map(|v| v.into_iter().map(Into::into).collect()),
    }
}

fn chunk_from_wire(c: ChunkWire, footer: &mut Vec<u8>) -> CompactColumnChunk {
    let statistics = c.statistics.map(|s| {
        let min_value = s.min_value.map(|bytes| append_to_footer(footer, &bytes));
        let max_value = s.max_value.map(|bytes| append_to_footer(footer, &bytes));
        CompactStatistics {
            null_count: s.null_count,
            distinct_count: None,
            max_value,
            min_value,
            is_max_value_exact: None,
            is_min_value_exact: None,
        }
    });

    // Fields dropped from the wire form (no read-path consumers in
    // polars; reconstruct as None/0 to keep the in-memory shape stable).
    // `chunk_offset` is the resolved start (= `dict_offset.unwrap_or(data_offset)`);
    // we set data_page_offset = chunk_offset and dict_offset = None so
    // `byte_range()` computes [chunk_offset, chunk_offset + chunk_size).
    let meta_data = CompactColumnMetaData {
        codec: c.codec,
        num_values: c.num_values,
        total_uncompressed_size: 0,
        total_compressed_size: c.chunk_size,
        data_page_offset: c.chunk_offset,
        index_page_offset: None,
        dictionary_page_offset: None,
        statistics,
        bloom_filter_offset: None,
        bloom_filter_length: None,
    };

    CompactColumnChunk {
        meta_data,
        offset_index_offset: None,
        offset_index_length: None,
        column_index_offset: None,
        column_index_length: None,
    }
}

fn append_to_footer(footer: &mut Vec<u8>, bytes: &[u8]) -> ByteRange {
    let offset = footer.len() as u32;
    footer.extend_from_slice(bytes);
    ByteRange {
        offset,
        len: bytes.len() as u32,
    }
}
