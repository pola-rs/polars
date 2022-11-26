///

#[must_use]
pub struct DeltaWriter<W> {
    writer: W,
    compression: write::CompressionOptions,
    statistics: bool,
    row_group_size: Option<usize>,
}