use super::*;

impl ALogicalPlan {
    /// Get the schema of the logical plan node but don't take projections into account at the scan
    /// level. This ensures we can apply the predicate
    pub(crate) fn scan_schema(&self) -> &SchemaRef {
        use ALogicalPlan::*;
        match self {
            Scan { file_info, .. } => &file_info.schema,
            #[cfg(feature = "python")]
            PythonScan { options, .. } => &options.schema,
            _ => unreachable!(),
        }
    }

    pub fn name(&self) -> &'static str {
        use ALogicalPlan::*;
        match self {
            Scan { scan_type, .. } => scan_type.into(),
            #[cfg(feature = "python")]
            PythonScan { .. } => "python_scan",
            Slice { .. } => "slice",
            Selection { .. } => "selection",
            DataFrameScan { .. } => "df",
            Projection { .. } => "projection",
            Sort { .. } => "sort",
            Cache { .. } => "cache",
            Aggregate { .. } => "aggregate",
            Join { .. } => "join",
            HStack { .. } => "hstack",
            Distinct { .. } => "distinct",
            MapFunction { .. } => "map_function",
            Union { .. } => "union",
            HConcat { .. } => "hconcat",
            ExtContext { .. } => "ext_context",
            Sink { payload, .. } => match payload {
                SinkType::Memory => "sink (memory)",
                SinkType::File { .. } => "sink (file)",
                #[cfg(feature = "cloud")]
                SinkType::Cloud { .. } => "sink (cloud)",
            },
            SimpleProjection { .. } => "simple_projection",
            Invalid => "invalid",
        }
    }

    /// Get the schema of the logical plan node.
    pub fn schema<'a>(&'a self, arena: &'a Arena<ALogicalPlan>) -> Cow<'a, SchemaRef> {
        use ALogicalPlan::*;
        let schema = match self {
            #[cfg(feature = "python")]
            PythonScan { options, .. } => options.output_schema.as_ref().unwrap_or(&options.schema),
            Union { inputs, .. } => return arena.get(inputs[0]).schema(arena),
            HConcat { schema, .. } => schema,
            Cache { input, .. } => return arena.get(*input).schema(arena),
            Sort { input, .. } => return arena.get(*input).schema(arena),
            Scan {
                output_schema,
                file_info,
                ..
            } => output_schema.as_ref().unwrap_or(&file_info.schema),
            DataFrameScan {
                schema,
                output_schema,
                ..
            } => output_schema.as_ref().unwrap_or(schema),
            Selection { input, .. } => return arena.get(*input).schema(arena),
            Projection { schema, .. } => schema,
            SimpleProjection { columns, .. } => columns,
            Aggregate { schema, .. } => schema,
            Join { schema, .. } => schema,
            HStack { schema, .. } => schema,
            Distinct { input, .. } | Sink { input, .. } => return arena.get(*input).schema(arena),
            Slice { input, .. } => return arena.get(*input).schema(arena),
            MapFunction { input, function } => {
                let input_schema = arena.get(*input).schema(arena);

                return match input_schema {
                    Cow::Owned(schema) => {
                        Cow::Owned(function.schema(&schema).unwrap().into_owned())
                    },
                    Cow::Borrowed(schema) => function.schema(schema).unwrap(),
                };
            },
            ExtContext { schema, .. } => schema,
            Invalid => unreachable!(),
        };
        Cow::Borrowed(schema)
    }
}
