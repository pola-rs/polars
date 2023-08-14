use polars_core::error::to_compute_err;

use super::*;
use crate::prelude::{AnonymousScan, AnonymousScanOptions, LazyJsonLineReader};

impl AnonymousScan for LazyJsonLineReader {
    fn name(&self) -> &'static str {
        "LazyJsonLineReader"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn scan(&self, scan_opts: AnonymousScanOptions) -> PolarsResult<DataFrame> {
        let schema = scan_opts.output_schema.unwrap_or(scan_opts.schema);
        JsonLineReader::from_path(&self.path)?
            .with_schema(schema)
            .with_rechunk(self.rechunk)
            .with_chunk_size(self.batch_size)
            .low_memory(self.low_memory)
            .with_n_rows(scan_opts.n_rows)
            .with_chunk_size(self.batch_size)
            .finish()
    }

    fn schema(&self, infer_schema_length: Option<usize>) -> PolarsResult<Schema> {
        let f = polars_utils::open_file(&self.path)?;
        let mut reader = std::io::BufReader::new(f);

        let data_type =
            polars_json::ndjson::infer(&mut reader, infer_schema_length).map_err(to_compute_err)?;
        let schema = Schema::from_iter(StructArray::get_fields(&data_type));

        Ok(schema)
    }
    fn allows_projection_pushdown(&self) -> bool {
        true
    }
}
#[cfg(test)]
mod test {
    use std::sync::Arc;

    use polars_arrow::error::{polars_bail, PolarsResult};
    use polars_plan::prelude::{AnonymousScan, FunctionRegistry, LogicalPlan};

    use crate::prelude::{LazyFileListReader, LazyFrame, LazyJsonLineReader};

    struct JsonEncodedRegistry;
    impl FunctionRegistry for JsonEncodedRegistry {
        fn try_encode_scan(
            &self,
            scan: &dyn AnonymousScan,
        ) -> PolarsResult<polars_plan::prelude::UserDefinedNode> {
            if scan.name() == "LazyJsonLineReader" {
                let lf = scan.as_any().downcast_ref::<LazyJsonLineReader>().unwrap();
                let bytes = serde_json::to_vec(lf).unwrap();
                Ok(polars_plan::prelude::UserDefinedNode {
                    name: "LazyJsonLineReader".to_string(),
                    bytes,
                })
            } else {
                polars_bail!(InvalidOperation: "cannot serialize scan")
            }
        }

        fn try_decode_scan(
            &self,
            node: &polars_plan::prelude::UserDefinedNode,
        ) -> PolarsResult<Option<Arc<dyn AnonymousScan>>> {
            if node.name == "LazyJsonLineReader" {
                let reader: LazyJsonLineReader = serde_json::from_slice(&node.bytes).unwrap();
                let scan = Arc::new(reader);
                Ok(Some(scan))
            } else {
                Ok(None)
            }
        }

        fn try_encode_udf(
            &self,
            _udf: &dyn polars_plan::prelude::DataFrameUdf,
        ) -> PolarsResult<polars_plan::prelude::UserDefinedNode> {
            todo!()
        }

        fn try_decode_udf(
            &self,
            _node: &polars_plan::prelude::UserDefinedNode,
        ) -> PolarsResult<Option<Arc<dyn polars_plan::prelude::DataFrameUdf>>> {
            todo!()
        }
    }

    #[test]
    fn test_serialize_plan() {
        let path = "path/to/some/file.json";
        let lf = LazyJsonLineReader::new(path).finish().unwrap().limit(1);

        let plan = lf.logical_plan;
        let mut buf = Vec::new();
        let mut serializer = serde_json::Serializer::new(&mut buf);

        plan.try_serialize(&mut serializer, &JsonEncodedRegistry)
            .unwrap();
        let mut deserializer = serde_json::Deserializer::from_slice(&buf);

        let deserialized =
            LogicalPlan::try_deserialize(&mut deserializer, &JsonEncodedRegistry).unwrap();
        println!("{:?}", deserialized);
        let lf: LazyFrame = deserialized.into();
        let df = lf.collect().unwrap();
        println!("{:?}", df);
    }
}
