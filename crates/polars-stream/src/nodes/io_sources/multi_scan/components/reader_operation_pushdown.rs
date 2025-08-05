use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_utils::slice_enum::Slice;

use crate::nodes::io_sources::multi_scan::components::row_deletions::ExternalFilterMask;
use crate::nodes::io_sources::multi_scan::pipeline::models::ExtraOperations;
use crate::nodes::io_sources::multi_scan::reader_interface::Projection;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;

/// Encapsulates logic for determining which operations to push into the underlying reader.
pub struct ReaderOperationPushdown<'a> {
    pub file_projection: Projection,
    pub reader_capabilities: ReaderCapabilities,
    pub external_filter_mask: Option<ExternalFilterMask>,
    /// Operations will be `take()`en out when pushed.
    pub extra_ops_post: &'a mut ExtraOperations,
}

impl ReaderOperationPushdown<'_> {
    pub fn push_operations(
        self,
    ) -> (
        Projection,
        Projection,
        Option<RowIndex>,
        Option<Slice>,
        Option<ScanIOPredicate>,
        Option<ExternalFilterMask>,
    ) {
        let Self {
            file_projection,
            reader_capabilities,
            external_filter_mask,
            extra_ops_post,
        } = self;

        use ReaderCapabilities as RC;

        let unsupported_external_filter_mask = external_filter_mask.is_some()
            && !reader_capabilities.contains(RC::EXTERNAL_FILTER_MASK);

        let unsupported_resolved_mapped_projection = match &file_projection {
            Projection::Plain(_) => false,
            Projection::Mapped { .. } => {
                !reader_capabilities.contains(RC::MAPPED_COLUMN_PROJECTION)
            },
        };

        let (projection_to_reader, projection_to_post) = if unsupported_resolved_mapped_projection {
            (file_projection.get_plain_pre_projection(), file_projection)
        } else {
            let mut projection_to_post = file_projection.clone();
            projection_to_post.clear_projection_transforms();
            (file_projection, projection_to_post)
        };

        // Notes
        // * If there is both a slice and deletions, DO NOT push deletions to the reader without
        //   pushing the slice.

        // If `unsupported_mapped_projection`, the file may contain a column sharing the name of
        // the row index column, but gets renamed by the column mapping.
        let row_index = if reader_capabilities.contains(RC::ROW_INDEX)
            && !(unsupported_resolved_mapped_projection || unsupported_external_filter_mask)
        {
            extra_ops_post.row_index.take()
        } else {
            None
        };

        let pre_slice = match &extra_ops_post.pre_slice {
            Some(Slice::Positive { .. }) if reader_capabilities.contains(RC::PRE_SLICE) => {
                extra_ops_post.pre_slice.take()
            },

            Some(Slice::Negative { .. })
                if reader_capabilities.contains(RC::NEGATIVE_PRE_SLICE) =>
            {
                extra_ops_post.pre_slice.take()
            },

            _ => None,
        };

        let push_predicate = !(!reader_capabilities.contains(RC::MAPPED_COLUMN_PROJECTION)
            || unsupported_external_filter_mask
            || extra_ops_post.predicate.is_none()
            || (extra_ops_post.row_index.is_some() || extra_ops_post.pre_slice.is_some())
            || !reader_capabilities.contains(RC::PARTIAL_FILTER));

        let mut predicate: Option<ScanIOPredicate> = None;

        if push_predicate {
            predicate = if reader_capabilities.contains(RC::FULL_FILTER) {
                // If the reader can fully handle the predicate itself, let it do it itself.
                extra_ops_post.predicate.take()
            } else {
                // Otherwise, we want to pass it and filter again afterwards.
                extra_ops_post.predicate.clone()
            }
        }

        (
            projection_to_reader,
            projection_to_post,
            row_index,
            pre_slice,
            predicate,
            external_filter_mask,
        )
    }
}
