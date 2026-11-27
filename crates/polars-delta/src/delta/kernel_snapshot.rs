use delta_kernel::history_manager::error::LogHistoryError;
use delta_kernel::path::LogPathFileType;
use delta_kernel::scan::{PartitionValuesOptions, StatsOptions};
use delta_kernel::schema::DataType;
use delta_kernel::{Engine, KernelError, Snapshot as KernelSnapshot, Version};
use polars_io_ext_ffi::common::{FfiError, FfiErrorKind};
use polars_io_ext_ffi::delta_v1::Request;
use serde_json::Value;
use url::Url;

use super::error::{Result, input, invalid, unsupported};
use super::host::Storage;
use super::kernel_engine::host_error;
use super::snapshot::Snapshot;
use super::{kernel_files, log, runtime};

pub(super) fn load(
    engine: &dyn Engine,
    storage: &Storage,
    root: Url,
    request: &Request,
) -> Result<Snapshot> {
    let build = |version: Option<Version>| {
        let mut builder = KernelSnapshot::builder_for(root.as_str());
        if let Some(version) = version {
            builder = builder.at_version(version);
        }
        builder.build(engine)
    };
    let requested = request
        .version
        .map(Version::try_from)
        .transpose()
        .map_err(|_| input("negative Delta version"))?;
    let initial = build(requested);
    let snapshot = if let Some(timestamp) = request.timestamp_ms {
        let latest = match &initial {
            Ok(snapshot) => Some((
                i64::try_from(snapshot.version()).map_err(invalid)?,
                metadata_and_protocol(snapshot)?.0,
            )),
            // Selecting an older version needs the latest history configuration, not
            // a successfully parsed latest schema. Read raw history metadata if schema
            // decoding fails.
            Err(error) if schema_decoding_error(error) => None,
            Err(error) => return Err(snapshot_error(error, requested)),
        };
        // Apply monotonic timestamp adjustment before comparing with the query, including
        // when no in-commit timestamps are present.
        let target = runtime::block_on(
            request.max_threads,
            log::time_travel_version(
                storage,
                request,
                timestamp,
                latest.as_ref().map(|(v, m)| (*v, m)),
            ),
        )?;
        let target = Version::try_from(target).map_err(invalid)?;
        match initial {
            Ok(snapshot) if snapshot.version() == target => snapshot,
            _ => build(Some(target)).map_err(|e| snapshot_error(&e, Some(target)))?,
        }
    } else {
        initial.map_err(|e| snapshot_error(&e, requested))?
    };
    let version = i64::try_from(snapshot.version()).map_err(invalid)?;
    let (metadata, protocol) = metadata_and_protocol(&snapshot)?;

    let mut sources = Vec::new();
    let mut files = Vec::new();
    if !request.schema_only {
        // JSON commits carry the original text stats. Their Boolean bounds and decimal
        // precision must reach the Polars parser before Kernel's typed coercions.
        let segment = snapshot.log_segment();
        let only_json_commits = segment.checkpoint_version.is_none()
            && segment.listed.ascending_compaction_files.is_empty()
            && segment
                .listed
                .ascending_commit_files
                .iter()
                .all(|file| file.file_type == LogPathFileType::Commit);
        let schema = snapshot.schema();
        let partition_columns = snapshot.table_configuration().logical_partition_columns();
        // Keep Kernel's indexed set for typed counts. Adding one extra column to the
        // expected shape can make Kernel reject every typed statistic as incompatible.
        let stats = if only_json_commits {
            StatsOptions::json_only()
        } else {
            StatsOptions::all_struct()
        };
        let zoned_checkpoint_partitions = segment.checkpoint_version.is_some()
            && partition_columns.iter().any(|name| {
                schema
                    .field(name)
                    .is_some_and(|f| *f.data_type() == DataType::TIMESTAMP)
            });
        let mut scan = snapshot
            .scan_builder()
            .with_stats(stats)
            .without_row_transforms();
        if zoned_checkpoint_partitions {
            scan = scan.with_partition_values(PartitionValuesOptions::with_struct());
        }
        let scan = scan.build().map_err(|e| kernel_error(&e))?;
        // With no predicate, even a later Add with updated stats must enter replay and mark its key.
        for batch in scan.scan_metadata(engine).map_err(|e| kernel_error(&e))? {
            kernel_files::append(
                batch.map_err(|e| kernel_error(&e))?,
                only_json_commits,
                &mut sources,
                &mut files,
            )?;
        }
        let mut keyed = files
            .iter()
            .map(|&(source, row)| (sources[source as usize].path(row as usize), (source, row)))
            .collect::<Vec<_>>();
        keyed.sort_unstable_by_key(|&(path, _)| path);
        if keyed.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(invalid("multiple live Delta adds for one path"));
        }
        files.clear();
        files.extend(keyed.into_iter().map(|(_, file)| file));
    }
    Ok(Snapshot {
        version,
        metadata,
        protocol,
        sources,
        files,
    })
}

fn metadata_and_protocol(snapshot: &KernelSnapshot) -> Result<(Value, Value)> {
    // In 0.30.1 these three accessors are internal-api. Switch to a public snapshot
    // accessor once Kernel exposes the protocol and raw metadata, including schemaString and id.
    let configuration = snapshot.table_configuration();
    let metadata = serde_json::to_value(configuration.metadata()).map_err(invalid)?;
    let protocol = serde_json::to_value(configuration.protocol()).map_err(invalid)?;
    Ok((metadata, protocol))
}

fn unsupported_schema(error: &KernelError) -> bool {
    match error {
        KernelError::Backtraced { source, .. } => unsupported_schema(source),
        KernelError::Schema(message) => message.starts_with("Unsupported Delta table type"),
        _ => false,
    }
}

fn schema_decoding_error(error: &KernelError) -> bool {
    match error {
        KernelError::Backtraced { source, .. } => schema_decoding_error(source),
        // This is the error from parsing the selected metadata's schemaString.
        KernelError::MalformedJson(_) => true,
        KernelError::Schema(message) => {
            unsupported_schema(error) || message.starts_with("Table schema is too deeply nested:")
        },
        _ => false,
    }
}

fn snapshot_error(error: &KernelError, requested: Option<Version>) -> FfiError {
    if matches!(error, KernelError::MissingVersion(missing) if Some(*missing) == requested) {
        return input(error);
    }
    kernel_error(error)
}

fn kernel_error(error: &KernelError) -> FfiError {
    if let Some(error) = host_error(error) {
        return error;
    }
    match error {
        KernelError::Backtraced { source, .. } => kernel_error(source),
        KernelError::Unsupported(_) => unsupported(error),
        KernelError::Schema(_) if unsupported_schema(error) => unsupported(error),
        KernelError::EmptyLog => input(error),
        KernelError::LogHistory(history) => match history.as_ref() {
            LogHistoryError::TimestampOutOfRange { .. }
            | LogHistoryError::NoCommitsFound { .. } => input(error),
            LogHistoryError::Internal {
                source: Some(source),
                ..
            } => kernel_error(source),
            _ => invalid(error),
        },
        KernelError::FileNotFound(_) => FfiError::new(FfiErrorKind::NOT_FOUND, error.to_string()),
        KernelError::IOError(_) => FfiError::new(FfiErrorKind::IO, error.to_string()),
        _ => invalid(error),
    }
}
