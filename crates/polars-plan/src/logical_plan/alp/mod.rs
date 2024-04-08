mod inputs;
mod schema;

use std::borrow::Cow;
use std::path::PathBuf;

use polars_core::prelude::*;
use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;

use super::projection_expr::*;
use crate::prelude::*;

/// [`ALogicalPlan`] is a representation of [`LogicalPlan`] with [`Node`]s which are allocated in an [`Arena`]
#[derive(Clone, Debug, Default)]
pub enum ALogicalPlan {
    #[cfg(feature = "python")]
    PythonScan {
        options: PythonOptions,
        predicate: Option<ExprIR>,
    },
    Slice {
        input: Node,
        offset: i64,
        len: IdxSize,
    },
    Selection {
        input: Node,
        predicate: ExprIR,
    },
    Scan {
        paths: Arc<[PathBuf]>,
        file_info: FileInfo,
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
        // schema of the projected file
        output_schema: Option<SchemaRef>,
        projection: Option<Arc<Vec<String>>>,
        selection: Option<ExprIR>,
    },
    // Only selects columns
    SimpleProjection {
        input: Node,
        columns: SchemaRef,
        duplicate_check: bool,
    },
    Projection {
        input: Node,
        expr: ProjectionExprs,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    Sort {
        input: Node,
        by_column: Vec<ExprIR>,
        args: SortArguments,
    },
    Cache {
        input: Node,
        // Unique ID.
        id: usize,
        /// How many hits the cache must be saved in memory.
        cache_hits: u32,
    },
    Aggregate {
        input: Node,
        keys: Vec<ExprIR>,
        aggs: Vec<ExprIR>,
        schema: SchemaRef,
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
        exprs: ProjectionExprs,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    Distinct {
        input: Node,
        options: DistinctOptions,
    },
    MapFunction {
        input: Node,
        function: FunctionNode,
    },
    Union {
        inputs: Vec<Node>,
        options: UnionOptions,
    },
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

#[cfg(test)]
mod test {
    use super::*;

    // skipped for now
    #[ignore]
    #[test]
    fn test_alp_size() {
        assert!(std::mem::size_of::<ALogicalPlan>() <= 152);
    }
}
