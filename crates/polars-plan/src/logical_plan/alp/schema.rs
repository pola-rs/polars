use super::*;

impl IR {
    /// Get the schema of the logical plan node but don't take projections into account at the scan
    /// level. This ensures we can apply the predicate
    pub(crate) fn scan_schema(&self) -> &SchemaRef {
        use IR::*;
        match self {
            Scan { file_info, .. } => &file_info.schema,
            #[cfg(feature = "python")]
            PythonScan { options, .. } => &options.schema,
            _ => unreachable!(),
        }
    }

    pub fn name(&self) -> &'static str {
        use IR::*;
        match self {
            Scan { scan_type, .. } => scan_type.into(),
            #[cfg(feature = "python")]
            PythonScan { .. } => "python_scan",
            Slice { .. } => "slice",
            Filter { .. } => "selection",
            DataFrameScan { .. } => "df",
            Select { .. } => "projection",
            Reduce { .. } => "reduce",
            Sort { .. } => "sort",
            Cache { .. } => "cache",
            GroupBy { .. } => "aggregate",
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

    pub fn input_schema<'a>(&'a self, arena: &'a Arena<IR>) -> Option<Cow<'a, SchemaRef>> {
        use IR::*;
        let schema = match self {
            #[cfg(feature = "python")]
            PythonScan { options, .. } => &options.schema,
            DataFrameScan { schema, .. } => schema,
            Scan { file_info, .. } => &file_info.schema,
            node => {
                let input = node.get_input()?;
                return Some(arena.get(input).schema(arena));
            },
        };
        Some(Cow::Borrowed(schema))
    }

    /// Get the schema of the logical plan node.
    pub fn schema<'a>(&'a self, arena: &'a Arena<IR>) -> Cow<'a, SchemaRef> {
        use IR::*;
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
            Filter { input, .. } => return arena.get(*input).schema(arena),
            Select { schema, .. } => schema,
            Reduce { schema, .. } => schema,
            SimpleProjection { columns, .. } => columns,
            GroupBy { schema, .. } => schema,
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
