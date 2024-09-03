mod dot;
mod format;
mod inputs;
mod schema;
pub(crate) mod tree_format;

use std::borrow::Cow;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub use dot::{EscapeLabel, IRDotDisplay, PathsDisplay};
pub use format::{ExprIRDisplay, IRDisplay};
use hive::HivePartitions;
use polars_core::prelude::*;
use polars_core::POOL;
use polars_utils::idx_vec::UnitVec;
use polars_utils::{format_pl_smallstr, unitvec};
#[cfg(feature = "ir_serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct IRPlan {
    pub lp_top: Node,
    pub lp_arena: Arena<IR>,
    pub expr_arena: Arena<AExpr>,
}

#[derive(Clone, Copy)]
pub struct IRPlanRef<'a> {
    pub lp_top: Node,
    pub lp_arena: &'a Arena<IR>,
    pub expr_arena: &'a Arena<AExpr>,
}

#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ScanSource {
    Files(Arc<[PathBuf]>),
    #[cfg_attr(feature = "ir_serde", serde(skip))]
    Buffer(Arc<[u8]>),
}

impl Default for ScanSource {
    fn default() -> Self {
        Self::Files(Arc::default())
    }
}

pub struct ScanSourceSliceInfo {
    pub item_slice: std::ops::Range<usize>,
    pub source_slice: std::ops::Range<usize>,
}

impl ScanSource {
    pub fn as_paths(&self) -> &[PathBuf] {
        match self {
            ScanSource::Files(paths) => paths,
            ScanSource::Buffer(_) => unimplemented!(),
        }
    }

    pub fn try_into_paths(&self) -> PolarsResult<Arc<[PathBuf]>> {
        match self {
            ScanSource::Files(paths) => Ok(paths.clone()),
            ScanSource::Buffer(_) => Err(polars_err!(
                nyi = "Unable to convert BytesIO scan into path"
            )),
        }
    }

    pub fn into_paths(&self) -> Arc<[PathBuf]> {
        match self {
            ScanSource::Files(paths) => paths.clone(),
            ScanSource::Buffer(_) => unimplemented!(),
        }
    }

    pub fn to_dsl(self, is_expanded: bool) -> DslScanSource {
        match self {
            ScanSource::Files(paths) => {
                DslScanSource::File(Arc::new(Mutex::new(ScanFileSource { paths, is_expanded })))
            },
            ScanSource::Buffer(buffer) => DslScanSource::Buffer(buffer),
        }
    }

    pub fn num_sources(&self) -> usize {
        match self {
            ScanSource::Files(paths) => paths.len(),
            ScanSource::Buffer(_) => 1,
        }
    }

    pub fn is_cloud_url(&self) -> PolarsResult<bool> {
        match self {
            ScanSource::Files(paths) => {
                Ok(polars_io::is_cloud_url(paths.first().ok_or_else(
                    || polars_err!(ComputeError: "expected at least 1 path"),
                )?))
            },
            ScanSource::Buffer(_) => Ok(false),
        }
    }

    pub fn id(&self) -> PlSmallStr {
        match self {
            ScanSource::Files(paths) if paths.is_empty() => PlSmallStr::from_static("EMPTY"),
            ScanSource::Files(paths) => PlSmallStr::from_str(paths[0].to_string_lossy().as_ref()),
            ScanSource::Buffer(_) => PlSmallStr::from_static("IN_MEMORY"),
        }
    }

    /// Normalize the slice and collect information as to what rows and parts of the source are
    /// used in this slice.
    pub fn collect_slice_information(
        &self,
        slice: (i64, usize),
        path_to_num_rows: impl Fn(&Path) -> PolarsResult<usize> + Send + Sync,
        buffer_to_num_rows: impl Fn(&[u8]) -> PolarsResult<usize> + Send + Sync,
    ) -> PolarsResult<ScanSourceSliceInfo> {
        fn slice_to_start_end(
            offset: i64,
            length: usize,
            num_rows: usize,
        ) -> std::ops::Range<usize> {
            if offset < 0 {
                let slice_start_as_n_from_end = -offset as usize;
                let (start, len) = if slice_start_as_n_from_end > num_rows {
                    // We need to trim the slice, e.g. SLICE[offset: -100, len: 75] on a file of 50
                    // rows should only give the first 25 rows.
                    let start_position = slice_start_as_n_from_end - num_rows;
                    (0, length.saturating_sub(start_position))
                } else {
                    (num_rows - slice_start_as_n_from_end, length)
                };

                let end = start.saturating_add(len);

                start..end
            } else {
                let offset = offset as usize;
                offset.min(num_rows)..(offset + length).min(num_rows)
            }
        }

        let (offset, length) = slice;

        Ok(match self {
            ScanSource::Files(paths) if paths.len() == 1 => {
                let num_rows = path_to_num_rows(&paths[0])?;
                ScanSourceSliceInfo {
                    item_slice: slice_to_start_end(offset, length, num_rows),
                    source_slice: 0..1,
                }
            },
            ScanSource::Files(paths) => {
                use rayon::prelude::*;

                assert_ne!(paths.len(), 0);

                // Walk the files in reverse until we find the first file, and then translate the
                // slice into a positive-offset equivalent.
                const CHUNK_SIZE: usize = 8;
                let mut row_counts = Vec::with_capacity(paths.len());

                POOL.install(|| {
                    for idx_end in (0..paths.len()).step_by(CHUNK_SIZE) {
                        let idx_start = idx_end.saturating_sub(CHUNK_SIZE);

                        row_counts.extend(
                            (idx_start..=idx_end)
                                .into_par_iter()
                                .map(|i| path_to_num_rows(&paths[i]))
                                .collect::<PolarsResult<Vec<_>>>()?
                                .into_iter()
                                .rev(),
                        );
                    }

                    PolarsResult::Ok(())
                })?;

                let num_rows = row_counts.iter().sum::<usize>();

                let item_slice = slice_to_start_end(offset, length, num_rows);

                let mut source_start = paths.len() - 1;
                let mut source_end = 0;

                let mut sum = 0;
                for (i, row_count) in row_counts.iter().rev().enumerate() {
                    if sum < item_slice.end {
                        source_end = usize::max(source_end, i);
                    }

                    sum += row_count;

                    if sum >= item_slice.start {
                        source_start = usize::min(source_start, i);
                    }
                }

                let source_slice = source_start..source_end + 1;

                ScanSourceSliceInfo {
                    item_slice,
                    source_slice,
                }
            },
            ScanSource::Buffer(buffer) => {
                let num_rows = buffer_to_num_rows(buffer)?;

                ScanSourceSliceInfo {
                    item_slice: slice_to_start_end(offset, length, num_rows),
                    source_slice: 0..1,
                }
            },
        })
    }
}

/// [`IR`] is a representation of [`DslPlan`] with [`Node`]s which are allocated in an [`Arena`]
/// In this IR the logical plan has access to the full dataset.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub enum IR {
    #[cfg(feature = "python")]
    PythonScan {
        options: PythonOptions,
    },
    Slice {
        input: Node,
        offset: i64,
        len: IdxSize,
    },
    Filter {
        input: Node,
        predicate: ExprIR,
    },
    Scan {
        sources: ScanSource,
        file_info: FileInfo,
        hive_parts: Option<Arc<Vec<HivePartitions>>>,
        predicate: Option<ExprIR>,
        /// schema of the projected file
        output_schema: Option<SchemaRef>,
        scan_type: FileScan,
        /// generic options that can be used for all file types.
        file_options: FileScanOptions,
    },
    DataFrameScan {
        df: Arc<DataFrame>,
        schema: SchemaRef,
        // Schema of the projected file
        // If `None`, no projection is applied
        output_schema: Option<SchemaRef>,
        // Predicate to apply on the DataFrame
        // All the columns required for the predicate are projected.
        filter: Option<ExprIR>,
    },
    // Only selects columns (semantically only has row access).
    // This is a more restricted operation than `Select`.
    SimpleProjection {
        input: Node,
        columns: SchemaRef,
    },
    // Special case of `select` where all operations reduce to a single row.
    Reduce {
        input: Node,
        exprs: Vec<ExprIR>,
        schema: SchemaRef,
    },
    // Polars' `select` operation. This may access full materialized data.
    Select {
        input: Node,
        expr: Vec<ExprIR>,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    Sort {
        input: Node,
        by_column: Vec<ExprIR>,
        slice: Option<(i64, usize)>,
        sort_options: SortMultipleOptions,
    },
    Cache {
        input: Node,
        // Unique ID.
        id: usize,
        /// How many hits the cache must be saved in memory.
        cache_hits: u32,
    },
    GroupBy {
        input: Node,
        keys: Vec<ExprIR>,
        aggs: Vec<ExprIR>,
        schema: SchemaRef,
        #[cfg_attr(feature = "ir_serde", serde(skip))]
        apply: Option<Arc<dyn DataFrameUdf>>,
        maintain_order: bool,
        options: Arc<GroupbyOptions>,
    },
    Join {
        input_left: Node,
        input_right: Node,
        schema: SchemaRef,
        left_on: Vec<ExprIR>,
        right_on: Vec<ExprIR>,
        options: Arc<JoinOptions>,
    },
    HStack {
        input: Node,
        exprs: Vec<ExprIR>,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    Distinct {
        input: Node,
        options: DistinctOptionsIR,
    },
    MapFunction {
        input: Node,
        function: FunctionIR,
    },
    Union {
        inputs: Vec<Node>,
        options: UnionOptions,
    },
    /// Horizontal concatenation
    /// - Invariant: the names will be unique
    HConcat {
        inputs: Vec<Node>,
        schema: SchemaRef,
        options: HConcatOptions,
    },
    ExtContext {
        input: Node,
        contexts: Vec<Node>,
        schema: SchemaRef,
    },
    Sink {
        input: Node,
        payload: SinkType,
    },
    #[default]
    Invalid,
}

impl IRPlan {
    pub fn new(top: Node, ir_arena: Arena<IR>, expr_arena: Arena<AExpr>) -> Self {
        Self {
            lp_top: top,
            lp_arena: ir_arena,
            expr_arena,
        }
    }

    pub fn root(&self) -> &IR {
        self.lp_arena.get(self.lp_top)
    }

    pub fn as_ref(&self) -> IRPlanRef {
        IRPlanRef {
            lp_top: self.lp_top,
            lp_arena: &self.lp_arena,
            expr_arena: &self.expr_arena,
        }
    }

    /// Extract the original logical plan if the plan is for the Streaming Engine
    pub fn extract_streaming_plan(&self) -> Option<IRPlanRef> {
        self.as_ref().extract_streaming_plan()
    }

    pub fn describe(&self) -> String {
        self.as_ref().describe()
    }

    pub fn describe_tree_format(&self) -> String {
        self.as_ref().describe_tree_format()
    }

    pub fn display(&self) -> format::IRDisplay {
        self.as_ref().display()
    }

    pub fn display_dot(&self) -> dot::IRDotDisplay {
        self.as_ref().display_dot()
    }
}

impl<'a> IRPlanRef<'a> {
    pub fn root(self) -> &'a IR {
        self.lp_arena.get(self.lp_top)
    }

    pub fn with_root(self, root: Node) -> Self {
        Self {
            lp_top: root,
            lp_arena: self.lp_arena,
            expr_arena: self.expr_arena,
        }
    }

    /// Extract the original logical plan if the plan is for the Streaming Engine
    pub fn extract_streaming_plan(self) -> Option<IRPlanRef<'a>> {
        // @NOTE: the streaming engine replaces the whole tree with a MapFunction { Pipeline, .. }
        // and puts the original plan somewhere in there. This is how we extract it. Disgusting, I
        // know.
        let IR::MapFunction { input: _, function } = self.root() else {
            return None;
        };

        let FunctionIR::Pipeline { original, .. } = function else {
            return None;
        };

        Some(original.as_ref()?.as_ref().as_ref())
    }

    pub fn display(self) -> format::IRDisplay<'a> {
        format::IRDisplay::new(self)
    }

    pub fn display_dot(self) -> dot::IRDotDisplay<'a> {
        dot::IRDotDisplay::new(self)
    }

    pub fn describe(self) -> String {
        self.display().to_string()
    }

    pub fn describe_tree_format(self) -> String {
        let mut visitor = tree_format::TreeFmtVisitor::default();
        tree_format::TreeFmtNode::root_logical_plan(self).traverse(&mut visitor);
        format!("{visitor:#?}")
    }
}

impl fmt::Debug for IRPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <format::IRDisplay as fmt::Display>::fmt(&self.display(), f)
    }
}

impl fmt::Debug for IRPlanRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <format::IRDisplay as fmt::Display>::fmt(&self.display(), f)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // skipped for now
    #[ignore]
    #[test]
    fn test_alp_size() {
        assert!(std::mem::size_of::<IR>() <= 152);
    }
}
