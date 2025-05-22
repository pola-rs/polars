use std::sync::Arc;

use polars_core::prelude::*;
#[cfg(feature = "csv")]
use polars_io::csv::read::CsvReadOptions;
#[cfg(feature = "ipc")]
use polars_io::ipc::IpcScanOptions;
#[cfg(feature = "parquet")]
use polars_io::parquet::read::ParquetOptions;

#[cfg(feature = "python")]
use crate::dsl::python_dsl::PythonFunction;
use crate::prelude::*;

pub struct DslBuilder(pub DslPlan);

impl From<DslPlan> for DslBuilder {
    fn from(lp: DslPlan) -> Self {
        DslBuilder(lp)
    }
}

impl DslBuilder {
    pub fn anonymous_scan(
        function: Arc<dyn AnonymousScan>,
        options: AnonymousScanOptions,
        unified_scan_args: UnifiedScanArgs,
    ) -> PolarsResult<Self> {
        let schema = unified_scan_args.schema.clone().ok_or_else(|| {
            polars_err!(
                ComputeError:
                "anonymous scan requires schema to be specified in unified_scan_args"
            )
        })?;

        Ok(DslPlan::Scan {
            sources: ScanSources::default(),
            file_info: Some(FileInfo {
                schema: schema.clone(),
                reader_schema: Some(either::Either::Right(schema)),
                ..Default::default()
            }),
            unified_scan_args: Box::new(unified_scan_args),
            scan_type: Box::new(FileScan::Anonymous {
                function,
                options: Arc::new(options),
            }),
            cached_ir: Default::default(),
        }
        .into())
    }

    #[cfg(feature = "parquet")]
    #[allow(clippy::too_many_arguments)]
    pub fn scan_parquet(
        sources: ScanSources,
        options: ParquetOptions,
        unified_scan_args: UnifiedScanArgs,
    ) -> PolarsResult<Self> {
        Ok(DslPlan::Scan {
            sources,
            file_info: None,
            unified_scan_args: Box::new(unified_scan_args),
            scan_type: Box::new(FileScan::Parquet {
                options,
                metadata: None,
            }),
            cached_ir: Default::default(),
        }
        .into())
    }

    #[cfg(feature = "ipc")]
    #[allow(clippy::too_many_arguments)]
    pub fn scan_ipc(
        sources: ScanSources,
        options: IpcScanOptions,
        unified_scan_args: UnifiedScanArgs,
    ) -> PolarsResult<Self> {
        Ok(DslPlan::Scan {
            sources,
            file_info: None,
            unified_scan_args: Box::new(unified_scan_args),
            scan_type: Box::new(FileScan::Ipc {
                options,
                metadata: None,
            }),
            cached_ir: Default::default(),
        }
        .into())
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "csv")]
    pub fn scan_csv(
        sources: ScanSources,
        options: CsvReadOptions,
        unified_scan_args: UnifiedScanArgs,
    ) -> PolarsResult<Self> {
        Ok(DslPlan::Scan {
            sources,
            file_info: None,
            unified_scan_args: Box::new(unified_scan_args),
            scan_type: Box::new(FileScan::Csv { options }),
            cached_ir: Default::default(),
        }
        .into())
    }

    #[cfg(feature = "python")]
    pub fn scan_python_dataset(
        dataset_object: polars_utils::python_function::PythonObject,
    ) -> DslBuilder {
        use super::python_dataset::PythonDatasetProvider;

        DslPlan::Scan {
            sources: ScanSources::default(),
            file_info: None,
            unified_scan_args: Default::default(),
            scan_type: Box::new(FileScan::PythonDataset {
                dataset_object: Arc::new(PythonDatasetProvider::new(dataset_object)),
                cached_ir: Default::default(),
            }),
            cached_ir: Default::default(),
        }
        .into()
    }

    pub fn cache(self) -> Self {
        let input = Arc::new(self.0);
        let id = input.as_ref() as *const DslPlan as usize;
        DslPlan::Cache { input, id }.into()
    }

    pub fn drop(self, to_drop: Vec<Selector>, strict: bool) -> Self {
        self.map_private(DslFunction::Drop(DropFunction { to_drop, strict }))
    }

    pub fn project(self, exprs: Vec<Expr>, options: ProjectionOptions) -> Self {
        DslPlan::Select {
            expr: exprs,
            input: Arc::new(self.0),
            options,
        }
        .into()
    }

    pub fn fill_null(self, fill_value: Expr) -> Self {
        self.project(
            vec![all().fill_null(fill_value)],
            ProjectionOptions {
                duplicate_check: false,
                ..Default::default()
            },
        )
    }

    pub fn drop_nans(self, subset: Option<Vec<Expr>>) -> Self {
        let is_nan = match subset {
            Some(subset) if subset.is_empty() => return self,
            Some(subset) => subset.into_iter().map(Expr::is_nan).collect(),
            None => vec![dtype_cols([DataType::Float32, DataType::Float64]).is_nan()],
        };
        self.remove(any_horizontal(is_nan).unwrap())
    }

    pub fn drop_nulls(self, subset: Option<Vec<Expr>>) -> Self {
        let is_not_null = match subset {
            Some(subset) if subset.is_empty() => return self,
            Some(subset) => subset.into_iter().map(Expr::is_not_null).collect(),
            None => vec![all().is_not_null()],
        };
        self.filter(all_horizontal(is_not_null).unwrap())
    }

    pub fn fill_nan(self, fill_value: Expr) -> Self {
        self.map_private(DslFunction::FillNan(fill_value))
    }

    pub fn with_columns(self, exprs: Vec<Expr>, options: ProjectionOptions) -> Self {
        if exprs.is_empty() {
            return self;
        }

        DslPlan::HStack {
            input: Arc::new(self.0),
            exprs,
            options,
        }
        .into()
    }

    pub fn match_to_schema(
        self,
        match_schema: SchemaRef,
        per_column: Arc<[MatchToSchemaPerColumn]>,
        extra_columns: ExtraColumnsPolicy,
    ) -> Self {
        DslPlan::MatchToSchema {
            input: Arc::new(self.0),
            match_schema,
            per_column,
            extra_columns,
        }
        .into()
    }

    pub fn with_context(self, contexts: Vec<DslPlan>) -> Self {
        DslPlan::ExtContext {
            input: Arc::new(self.0),
            contexts,
        }
        .into()
    }

    /// Apply a filter predicate, keeping the rows that match it.
    pub fn filter(self, predicate: Expr) -> Self {
        DslPlan::Filter {
            predicate,
            input: Arc::new(self.0),
        }
        .into()
    }

    /// Remove rows matching a filter predicate (note that rows
    /// where the predicate resolves to `null` are *not* removed).
    pub fn remove(self, predicate: Expr) -> Self {
        DslPlan::Filter {
            predicate: predicate.neq_missing(lit(true)),
            input: Arc::new(self.0),
        }
        .into()
    }

    pub fn group_by<E: AsRef<[Expr]>>(
        self,
        keys: Vec<Expr>,
        aggs: E,
        apply: Option<(Arc<dyn DataFrameUdf>, SchemaRef)>,
        maintain_order: bool,
        #[cfg(feature = "dynamic_group_by")] dynamic_options: Option<DynamicGroupOptions>,
        #[cfg(feature = "dynamic_group_by")] rolling_options: Option<RollingGroupOptions>,
    ) -> Self {
        let aggs = aggs.as_ref().to_vec();
        let options = GroupbyOptions {
            #[cfg(feature = "dynamic_group_by")]
            dynamic: dynamic_options,
            #[cfg(feature = "dynamic_group_by")]
            rolling: rolling_options,
            slice: None,
        };

        DslPlan::GroupBy {
            input: Arc::new(self.0),
            keys,
            aggs,
            apply,
            maintain_order,
            options: Arc::new(options),
        }
        .into()
    }

    pub fn build(self) -> DslPlan {
        self.0
    }

    pub fn from_existing_df(df: DataFrame) -> Self {
        let schema = df.schema().clone();
        DslPlan::DataFrameScan {
            df: Arc::new(df),
            schema,
        }
        .into()
    }

    pub fn sort(self, by_column: Vec<Expr>, sort_options: SortMultipleOptions) -> Self {
        DslPlan::Sort {
            input: Arc::new(self.0),
            by_column,
            slice: None,
            sort_options,
        }
        .into()
    }

    pub fn explode(self, columns: Vec<Selector>, allow_empty: bool) -> Self {
        DslPlan::MapFunction {
            input: Arc::new(self.0),
            function: DslFunction::Explode {
                columns,
                allow_empty,
            },
        }
        .into()
    }

    #[cfg(feature = "pivot")]
    pub fn unpivot(self, args: UnpivotArgsDSL) -> Self {
        DslPlan::MapFunction {
            input: Arc::new(self.0),
            function: DslFunction::Unpivot { args },
        }
        .into()
    }

    pub fn row_index(self, name: PlSmallStr, offset: Option<IdxSize>) -> Self {
        DslPlan::MapFunction {
            input: Arc::new(self.0),
            function: DslFunction::RowIndex { name, offset },
        }
        .into()
    }

    pub fn distinct(self, options: DistinctOptionsDSL) -> Self {
        DslPlan::Distinct {
            input: Arc::new(self.0),
            options,
        }
        .into()
    }

    pub fn slice(self, offset: i64, len: IdxSize) -> Self {
        DslPlan::Slice {
            input: Arc::new(self.0),
            offset,
            len,
        }
        .into()
    }

    pub fn join(
        self,
        other: DslPlan,
        left_on: Vec<Expr>,
        right_on: Vec<Expr>,
        options: Arc<JoinOptions>,
    ) -> Self {
        DslPlan::Join {
            input_left: Arc::new(self.0),
            input_right: Arc::new(other),
            left_on,
            right_on,
            predicates: Default::default(),
            options,
        }
        .into()
    }
    pub fn map_private(self, function: DslFunction) -> Self {
        DslPlan::MapFunction {
            input: Arc::new(self.0),
            function,
        }
        .into()
    }

    #[cfg(feature = "python")]
    pub fn map_python(
        self,
        function: PythonFunction,
        optimizations: AllowedOptimizations,
        schema: Option<SchemaRef>,
        validate_output: bool,
    ) -> Self {
        DslPlan::MapFunction {
            input: Arc::new(self.0),
            function: DslFunction::OpaquePython(OpaquePythonUdf {
                function,
                schema,
                predicate_pd: optimizations.contains(OptFlags::PREDICATE_PUSHDOWN),
                projection_pd: optimizations.contains(OptFlags::PROJECTION_PUSHDOWN),
                streamable: optimizations.contains(OptFlags::STREAMING),
                validate_output,
            }),
        }
        .into()
    }

    pub fn map<F>(
        self,
        function: F,
        optimizations: AllowedOptimizations,
        schema: Option<Arc<dyn UdfSchema>>,
        name: PlSmallStr,
    ) -> Self
    where
        F: DataFrameUdf + 'static,
    {
        let function = Arc::new(function);

        DslPlan::MapFunction {
            input: Arc::new(self.0),
            function: DslFunction::FunctionIR(FunctionIR::Opaque {
                function,
                schema,
                predicate_pd: optimizations.contains(OptFlags::PREDICATE_PUSHDOWN),
                projection_pd: optimizations.contains(OptFlags::PROJECTION_PUSHDOWN),
                streamable: optimizations.contains(OptFlags::STREAMING),
                fmt_str: name,
            }),
        }
        .into()
    }
}
