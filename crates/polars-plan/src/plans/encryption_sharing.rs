//! Sharing of the Parquet encryption and decryption configurations in a plan.
//!
//! Configurations with the same ID are separate objects after a plan is deserialized.
//! Sharing them again means they share caches, such as the KMS clients and keys of a
//! crypto factory.

use polars_io::parquet::encryption::EncryptionSharing;
use polars_io::parquet::read::ParquetOptions;

use super::*;

/// Uses the shared decryption configuration with the same ID.
pub(crate) fn share_decryption(options: &mut ParquetOptions, sharing: &mut EncryptionSharing) {
    options.decryption = options
        .decryption
        .take()
        .map(|decryption| sharing.share_decryption(decryption));
}

/// Uses the shared encryption configuration with the same ID.
pub(crate) fn share_encryption(file_format: &mut FileWriteFormat, sharing: &mut EncryptionSharing) {
    if let FileWriteFormat::Parquet(options) = file_format
        && options.encryption.is_some()
    {
        let options = Arc::make_mut(options);
        options.encryption = options
            .encryption
            .take()
            .map(|encryption| sharing.share_encryption(encryption));
    }
}

impl IRPlan {
    /// Shares the Parquet encryption and decryption configurations in this plan that
    /// have the same ID.
    ///
    /// Call this after deserializing an `IRPlan`, so that configurations that were
    /// shared before serialization share caches again. Plans converted from a
    /// [`DslPlan`] are already shared.
    pub fn share_encryption(&mut self) {
        let mut sharing = EncryptionSharing::default();
        for i in 0..self.lp_arena.len() {
            match self.lp_arena.get_mut(Node(i)) {
                IR::Scan { scan_type, .. } => {
                    if let FileScanIR::Parquet { options, .. } = scan_type.as_mut() {
                        share_decryption(options, &mut sharing);
                    }
                },
                IR::Sink { payload, .. } => match payload {
                    SinkTypeIR::File(options) => {
                        share_encryption(&mut options.file_format, &mut sharing)
                    },
                    SinkTypeIR::Partitioned(options) => {
                        share_encryption(&mut options.file_format, &mut sharing)
                    },
                    SinkTypeIR::Memory | SinkTypeIR::Callback(_) => {},
                },
                _ => {},
            }
        }
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use polars_io::parquet::encryption::ParquetEncryption;
    use polars_io::parquet::write::ParquetWriteOptions;
    use polars_parquet::parquet::encryption::encrypt::FileEncryptionProperties;
    use polars_utils::pl_path::PlRefPath;
    use polars_utils::pl_serialize::{deserialize_dsl, serialize_dsl};

    use super::*;

    /// Whether two encryptions are the same object, using the fixed properties they
    /// hold.
    fn same_object(a: &ParquetEncryption, b: &ParquetEncryption) -> bool {
        Arc::ptr_eq(&a.file_properties().unwrap(), &b.file_properties().unwrap())
    }

    fn sink_encryption(plan: &IRPlan, node: Node) -> &ParquetEncryption {
        let IR::Sink {
            payload: SinkTypeIR::File(options),
            ..
        } = plan.lp_arena.get(node)
        else {
            unreachable!()
        };
        let FileWriteFormat::Parquet(options) = &options.file_format else {
            unreachable!()
        };
        options.encryption.as_ref().unwrap()
    }

    fn file_sink(input: Node, encryption: ParquetEncryption) -> IR {
        IR::Sink {
            input,
            payload: SinkTypeIR::File(FileSinkOptions {
                target: SinkTarget::Path(PlRefPath::new("out.parquet")),
                file_format: FileWriteFormat::Parquet(Arc::new(ParquetWriteOptions {
                    encryption: Some(encryption),
                    ..Default::default()
                })),
                unified_sink_args: Default::default(),
            }),
        }
    }

    #[test]
    fn test_share_encryption() {
        let properties = FileEncryptionProperties::builder(b"0123456789012345".to_vec())
            .build()
            .unwrap();
        let encryption = ParquetEncryption::from(properties);

        // Separate deserializations give separate objects with the same ID.
        let mut bytes = Vec::new();
        serialize_dsl(&mut bytes, &encryption).unwrap();
        let a: ParquetEncryption = deserialize_dsl(&mut bytes.as_slice()).unwrap();
        let b: ParquetEncryption = deserialize_dsl(&mut bytes.as_slice()).unwrap();
        assert!(!same_object(&a, &b));

        let mut lp_arena = Arena::new();
        let input = lp_arena.add(IR::DataFrameScan {
            df: Arc::new(Default::default()),
            schema: Arc::new(Default::default()),
            output_schema: None,
        });
        let sink_a = lp_arena.add(file_sink(input, a.clone()));
        let sink_b = lp_arena.add(file_sink(input, b));
        let top = lp_arena.add(IR::SinkMultiple {
            inputs: vec![sink_a, sink_b],
        });
        let mut plan = IRPlan::new(top, lp_arena, Arena::new());

        plan.share_encryption();

        assert!(same_object(sink_encryption(&plan, sink_a), &a));
        assert!(same_object(sink_encryption(&plan, sink_b), &a));
    }
}
