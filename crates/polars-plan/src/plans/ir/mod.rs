mod dot;
mod format;
mod inputs;
mod schema;
pub(crate) mod tree_format;

use std::borrow::Cow;
use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};

pub use dot::{EscapeLabel, IRDotDisplay, PathsDisplay, ScanSourcesDisplay};
pub use format::{ExprIRDisplay, IRDisplay};
use hive::HivePartitions;
use polars_core::error::feature_gated;
use polars_core::prelude::*;
use polars_utils::idx_vec::UnitVec;
use polars_utils::mmap::MemSlice;
use polars_utils::unitvec;
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub enum ScanSources {
    Paths(Arc<[PathBuf]>),

    #[cfg_attr(feature = "serde", serde(skip))]
    Files(Arc<[File]>),
    #[cfg_attr(feature = "serde", serde(skip))]
    Buffers(Arc<[bytes::Bytes]>),
}

impl std::hash::Hash for ScanSources {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);

        // @NOTE: This is a bit crazy
        match self {
            Self::Paths(paths) => paths.hash(state),
            Self::Files(files) => files.as_ptr().hash(state),
            Self::Buffers(buffers) => buffers.as_ptr().hash(state),
        }
    }
}

impl PartialEq for ScanSources {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ScanSources::Paths(l), ScanSources::Paths(r)) => l == r,
            _ => false,
        }
    }
}

impl Eq for ScanSources {}

#[derive(Debug, Clone, Copy)]
pub enum ScanSourceRef<'a> {
    Path(&'a Path),
    File(&'a File),
    Buffer(&'a bytes::Bytes),
}

pub struct ScanSourceSliceInfo {
    pub item_slice: std::ops::Range<usize>,
    pub source_slice: std::ops::Range<usize>,
}

impl Default for ScanSources {
    fn default() -> Self {
        Self::Buffers(Arc::default())
    }
}

impl<'a> ScanSourceRef<'a> {
    pub fn to_file_path(&self) -> &str {
        match self {
            Self::Path(path) => path.to_str().unwrap(),
            Self::File(_) => "open-file",
            Self::Buffer(_) => "in-mem",
        }
    }

    pub fn to_memslice(&self) -> PolarsResult<MemSlice> {
        self.to_memslice_possibly_async(false, None, 0)
    }

    pub fn to_memslice_async_latest(&self, run_async: bool) -> PolarsResult<MemSlice> {
        match self {
            ScanSourceRef::Path(path) => {
                let file = if run_async {
                    feature_gated!("cloud", {
                        polars_io::file_cache::FILE_CACHE
                            .get_entry(path.to_str().unwrap())
                            // Safety: This was initialized by schema inference.
                            .unwrap()
                            .try_open_assume_latest()?
                    })
                } else {
                    polars_utils::open_file(path)?
                };

                Ok(MemSlice::from_mmap(Arc::new(unsafe {
                    memmap::Mmap::map(&file)?
                })))
            },
            ScanSourceRef::File(file) => Ok(MemSlice::from_mmap(Arc::new(unsafe {
                memmap::Mmap::map(*file)?
            }))),
            ScanSourceRef::Buffer(buff) => Ok(MemSlice::from_bytes((*buff).clone())),
        }
    }

    pub fn to_memslice_possibly_async(
        &self,
        run_async: bool,
        #[cfg(feature = "cloud")] cache_entries: Option<
            &Vec<Arc<polars_io::file_cache::FileCacheEntry>>,
        >,
        #[cfg(not(feature = "cloud"))] cache_entries: Option<&()>,
        index: usize,
    ) -> PolarsResult<MemSlice> {
        match self {
            Self::Path(path) => {
                let f = if run_async {
                    feature_gated!("cloud", {
                        cache_entries.unwrap()[index].try_open_check_latest()?
                    })
                } else {
                    polars_utils::open_file(path)?
                };

                let mmap = unsafe { memmap::Mmap::map(&f)? };
                Ok(MemSlice::from_mmap(Arc::new(mmap)))
            },
            Self::File(file) => {
                let mmap = unsafe { memmap::Mmap::map(*file)? };
                Ok(MemSlice::from_mmap(Arc::new(mmap)))
            },
            Self::Buffer(buff) => Ok(MemSlice::from_bytes((*buff).clone())),
        }
    }
}

impl ScanSources {
    pub fn iter(&self) -> ScanSourceIter {
        ScanSourceIter {
            sources: self,
            offset: 0,
        }
    }

    pub fn as_paths(&self) -> Option<&[PathBuf]> {
        match self {
            Self::Paths(paths) => Some(paths.as_ref()),
            Self::Files(_) | Self::Buffers(_) => None,
        }
    }

    pub fn into_paths(&self) -> Option<Arc<[PathBuf]>> {
        match self {
            Self::Paths(paths) => Some(paths.clone()),
            Self::Files(_) | Self::Buffers(_) => None,
        }
    }

    pub fn first_path(&self) -> Option<&Path> {
        match self {
            Self::Paths(paths) => paths.first().map(|p| p.as_path()),
            Self::Files(_) | Self::Buffers(_) => None,
        }
    }

    pub fn to_dsl(self, is_expanded: bool) -> DslScanSources {
        DslScanSources {
            sources: self,
            is_expanded,
        }
    }

    pub fn is_files(&self) -> bool {
        matches!(self, Self::Paths(_))
    }

    pub fn is_cloud_url(&self) -> bool {
        match self {
            Self::Paths(paths) => paths.first().map_or(false, polars_io::is_cloud_url),
            Self::Files(_) | Self::Buffers(_) => false,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Paths(s) => s.len(),
            Self::Files(s) => s.len(),
            Self::Buffers(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn first(&self) -> Option<ScanSourceRef> {
        self.get(0)
    }

    pub fn id(&self) -> PlSmallStr {
        if self.is_empty() {
            return PlSmallStr::from_static("EMPTY");
        }

        match self {
            Self::Paths(paths) => {
                PlSmallStr::from_str(paths.first().unwrap().to_string_lossy().as_ref())
            },
            Self::Files(_) => PlSmallStr::from_static("OPEN_FILES"),
            Self::Buffers(_) => PlSmallStr::from_static("IN_MEMORY"),
        }
    }

    pub fn get(&self, idx: usize) -> Option<ScanSourceRef> {
        match self {
            Self::Paths(paths) => paths.get(idx).map(|p| ScanSourceRef::Path(p)),
            Self::Files(files) => files.get(idx).map(ScanSourceRef::File),
            Self::Buffers(buffers) => buffers.get(idx).map(ScanSourceRef::Buffer),
        }
    }

    pub fn at(&self, idx: usize) -> ScanSourceRef {
        self.get(idx).unwrap()
    }
}

pub struct ScanSourceIter<'a> {
    sources: &'a ScanSources,
    offset: usize,
}

impl<'a> Iterator for ScanSourceIter<'a> {
    type Item = ScanSourceRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = match self.sources {
            ScanSources::Paths(paths) => ScanSourceRef::Path(paths.get(self.offset)?),
            ScanSources::Files(files) => ScanSourceRef::File(files.get(self.offset)?),
            ScanSources::Buffers(buffers) => ScanSourceRef::Buffer(buffers.get(self.offset)?),
        };

        self.offset += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.sources.len() - self.offset;
        (len, Some(len))
    }
}

impl<'a> ExactSizeIterator for ScanSourceIter<'a> {}

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
        sources: ScanSources,
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
