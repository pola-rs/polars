use std::sync::Arc;

use bytes::Bytes;
use delta_kernel::arrow::array::{
    Array, ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray, StructArray,
};
use delta_kernel::arrow::compute::{CastOptions, cast, cast_with_options, filter};
use delta_kernel::arrow::datatypes::DataType as ArrowDataType;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::schema::{DataType, SchemaRef, StructField, StructType};
use delta_kernel::{
    EngineData, FileDataReadResultIterator, FileMeta, FileSize, FileSlice, FilteredEngineData,
    JsonHandler, KernelError, KernelResult, KernelResultIterator, KernelResultIteratorStatic,
    ParquetFooter, ParquetHandler, PredicateRef, StorageHandler,
};
use polars_utils::aliases::PlHashMap;
use url::Url;

use super::checkpoint_bounds::{CheckpointBounds, CheckpointLayout};
use super::{
    HostError, HostIo, MISSING_MODIFIED_MS, ROOT, closed, input, mask_crc, read_only, runtime,
    with_checkpoint_stats_text,
};

/// Commit files Kernel is expected to read next are fetched concurrently: one round trip per
/// window instead of one per file. The window starts small, as a schema-only read can stop at the
/// newest commits, and doubles up to the bound on objects held ahead of their reader. Listed
/// sizes also bound a window to a byte budget, though it always holds at least one file.
const PREFETCH_FIRST_WINDOW: usize = 4;
const PREFETCH_MAX_WINDOW: usize = 16;
const PREFETCH_WINDOW_BYTES: u64 = 32 << 20;

/// End of the window starting at `from`: at most `window` files and `budget` listed bytes, but
/// never fewer than one file.
pub(super) fn window_end(files: &[FileMeta], from: usize, window: usize, budget: u64) -> usize {
    let mut bytes = 0u64;
    let taken = files[from..]
        .iter()
        .take(window)
        .take_while(|file| {
            bytes = bytes.saturating_add(file.size);
            bytes <= budget || std::ptr::eq(*file, &files[from])
        })
        .count();
    from + taken
}

/// Fetch `files[from..]`'s next window of JSON log files in one host callback and park the
/// results for `HostStore::get_opts`. Returns the end of the window. Failures, and a closed
/// engine, are left to the ordinary read of that file.
fn prefetch_window(io: &HostIo, files: &[FileMeta], from: usize, window: usize) -> usize {
    let end = window_end(files, from, window, PREFETCH_WINDOW_BYTES);
    let urls = {
        let parked = io.prefetched.lock().unwrap();
        files[from..end]
            .iter()
            .filter(|file| file.location.path().ends_with(".json"))
            .filter_map(|file| io.metadata_path(&file.location).ok())
            // A parked failure is fetched again: a transient error must not outlive its pass.
            .filter(|url| !matches!(parked.get(url), Some(Ok(_))))
            .collect::<Vec<_>>()
    };
    if urls.len() > 1 {
        let results = runtime::block_on(None, async {
            let live = io.live.lock().await;
            match live.as_ref() {
                Some(live) => Some(live.get_many(&urls).await),
                None => None,
            }
        });
        if let Some(results) = results {
            io.prefetched
                .lock()
                .unwrap()
                .extend(urls.into_iter().zip(results));
        }
    }
    end
}

pub(super) struct HostStorage(pub(super) Arc<HostIo>);

impl StorageHandler for HostStorage {
    fn list_from(&self, location: &Url) -> KernelResult<KernelResultIteratorStatic<FileMeta>> {
        // The only recursive listing used by snapshot planning is the version-named region
        // of _delta_log. Direct-child listing covers that region; sidecars are opened by name.
        let after = self.0.relative_path(location)?;
        if after.contains('/') {
            return Err(KernelError::generic(
                "nested Delta log listing is unsupported",
            ));
        }
        let start = (!after.is_empty() && after.bytes().all(|b| b.is_ascii_digit()))
            .then(|| format!("{}{after}", self.0.prefix));
        let listed = runtime::block_on(None, async {
            let live = self.0.live.lock().await;
            let live = live.as_ref().ok_or_else(closed)?;
            live.storage
                .list_with_sizes(&self.0.prefix, start.as_deref())
                .await
        })
        .map_err(HostError::kernel)?;

        let parent = Url::parse(ROOT)?.join("_delta_log/")?;
        let mut files = Vec::with_capacity(listed.len());
        for (object, size) in listed {
            let name = object.url.strip_prefix(&self.0.prefix).ok_or_else(|| {
                KernelError::generic("host listed an object outside the Delta log")
            })?;
            if name.ends_with(".crc") {
                continue;
            }
            if name.is_empty() || name.contains('/') {
                return Err(KernelError::generic(
                    "host listed a non-child of the Delta log",
                ));
            }
            let mut url = parent.clone();
            url.path_segments_mut()
                .expect("Delta metadata root has a path")
                .pop_if_empty()
                .push(name);
            if url.as_str() > location.as_str() {
                files.push(FileMeta::new(
                    url,
                    object.modified_ms.unwrap_or(MISSING_MODIFIED_MS),
                    size,
                ));
            }
        }
        files.sort_unstable_by(|a, b| a.location.as_str().cmp(b.location.as_str()));
        Ok(Box::new(files.into_iter().map(Ok)))
    }

    fn read_files(&self, files: Vec<FileSlice>) -> KernelResult<KernelResultIteratorStatic<Bytes>> {
        let io = self.0.clone();
        Ok(Box::new(files.into_iter().map(move |(location, range)| {
            let url = io.metadata_path(&location)?;
            mask_crc(&url)?;
            runtime::block_on(None, async {
                let mut live = io.live.lock().await;
                let live = live.as_mut().ok_or_else(closed).map_err(HostError::from)?;
                match range {
                    Some(range) => live.range(&url, range).await,
                    None => live.get(&url).await,
                }
                .map(Bytes::from_owner)
            })
            .map_err(HostError::into_kernel)
        })))
    }

    fn head(&self, location: &Url) -> KernelResult<FileMeta> {
        let url = self.0.metadata_path(location)?;
        mask_crc(&url)?;
        let size = runtime::block_on(None, async {
            let mut live = self.0.live.lock().await;
            let live = live.as_mut().ok_or_else(closed).map_err(HostError::from)?;
            live.pin(&url, None).await
        })
        .map_err(HostError::into_kernel)?;
        Ok(FileMeta::new(location.clone(), MISSING_MODIFIED_MS, size))
    }

    fn copy_atomic(&self, _: &Url, _: &Url) -> KernelResult<()> {
        Err(read_only())
    }

    fn put(&self, _: &Url, _: Bytes, _: bool) -> KernelResult<()> {
        Err(read_only())
    }

    fn delete(&self, _: &Url) -> KernelResult<()> {
        Err(read_only())
    }
}

pub(super) struct HostJson(pub(super) Arc<HostIo>, pub(super) Arc<dyn JsonHandler>);

impl JsonHandler for HostJson {
    fn parse_json(
        &self,
        data: Box<dyn EngineData>,
        schema: SchemaRef,
    ) -> KernelResult<Box<dyn EngineData>> {
        self.1.parse_json(data, schema)
    }

    fn read_json_files(
        &self,
        files: &[FileMeta],
        schema: SchemaRef,
        predicate: Option<PredicateRef>,
    ) -> KernelResult<FileDataReadResultIterator> {
        // Kernel derives the commit version from the synthesized _file column. Keep
        // the synthetic log URL and its original basename through the JSON handler.
        for file in files {
            self.0.relative_path(&file.location)?;
            mask_crc(file.location.path())?;
        }
        let discover_sidecars = schema.fields().len() == 1 && schema.field("sidecar").is_some();
        let schema = with_checkpoint_stats_text(snapshot_projection(schema)?)?;
        let schema = host_sidecar_projection(schema, discover_sidecars, true)?;
        if discover_sidecars {
            let batches = self.1.read_json_files(files, schema, predicate)?;
            return discover_checkpoint_sidecars(&self.0, batches, true);
        }
        if schema.field("add").is_none() || schema.field("remove").is_none() {
            let (io, inner, all) = (self.0.clone(), self.1.clone(), files.to_vec());
            let (mut fetched, mut window) = (0, PREFETCH_FIRST_WINDOW);
            return Ok(Box::new((0..all.len()).flat_map(move |i| {
                let file = all[i].clone();
                if i >= fetched {
                    fetched = prefetch_window(&io, &all, i, window);
                    window = (window * 2).min(PREFETCH_MAX_WINDOW);
                }
                match inner.read_json_files(&[file], schema.clone(), predicate.clone()) {
                    Ok(batches) => batches,
                    Err(error) => Box::new(std::iter::once(Err(error))),
                }
            })));
        }

        // There is no commit-name column in the scan projection. Read one object at a
        // time so the check sees every batch of the same commit and resets at its end.
        let all = files.to_vec();
        let files = files.to_vec();
        let io = self.0.clone();
        let (mut fetched, mut window) = (0, PREFETCH_FIRST_WINDOW);
        let inner = self.1.clone();
        let original_columns = (0..schema.num_fields()).collect::<Arc<[_]>>();
        let commit_schema = input::commit_projection(&schema)?;
        Ok(Box::new(files.into_iter().enumerate().flat_map(
            move |(i, file)| {
                if i >= fetched {
                    fetched = prefetch_window(&io, &all, i, window);
                    window = (window * 2).min(PREFETCH_MAX_WINDOW);
                }
                let validate = file
                    .location
                    .path_segments()
                    .and_then(Iterator::last)
                    .is_some_and(input::is_commit);
                let schema = if validate { &commit_schema } else { &schema };
                match inner.read_json_files(&[file], schema.clone(), predicate.clone()) {
                    Ok(batches) if validate => {
                        let mut actions = input::CommitActions::default();
                        let original_columns = original_columns.clone();
                        Box::new(batches.map(move |batch| actions.check(batch?, &original_columns)))
                            as FileDataReadResultIterator
                    },
                    Ok(batches) => batches,
                    Err(error) => Box::new(std::iter::once(Err(error))),
                }
            },
        )))
    }

    fn write_json_file(
        &self,
        _: &Url,
        _: KernelResultIterator<'_, FilteredEngineData>,
        _: bool,
    ) -> KernelResult<FileSize> {
        Err(read_only())
    }
}

fn host_sidecar_projection(
    schema: SchemaRef,
    discover: bool,
    json: bool,
) -> KernelResult<SchemaRef> {
    let Some(sidecar) = schema.field("sidecar") else {
        return Ok(schema);
    };
    let DataType::Struct(children) = sidecar.data_type() else {
        return Ok(schema);
    };
    let mut children: Vec<_> = children.fields().cloned().collect();
    for field in &mut children {
        if field.name() == "modificationTime" && field.data_type == DataType::LONG {
            field.nullable = true;
        }
    }
    let mut fields: Vec<_> = schema.fields().cloned().collect();
    fields
        .iter_mut()
        .find(|f| f.name() == "sidecar")
        .unwrap()
        .data_type = StructType::try_new(children)?.into();
    if discover {
        let mut action = vec![StructField::nullable("path", DataType::STRING)];
        if json {
            // Kernel's checkpoint replay drops tombstones. Read only their keys here
            // before handing the augmented projection back to sidecar discovery.
            let dv = StructType::try_new([
                StructField::nullable("storageType", DataType::STRING),
                StructField::nullable("pathOrInlineDv", DataType::STRING),
                StructField::nullable("offset", DataType::INTEGER),
            ])?;
            action.push(StructField::nullable("deletionVector", dv));
        }
        let action = StructType::try_new(action)?;
        fields.extend([
            StructField::nullable("add", action.clone()),
            StructField::nullable("remove", action),
        ]);
    }
    Ok(Arc::new(StructType::try_new(fields)?))
}

fn discover_checkpoint_sidecars(
    io: &HostIo,
    batches: FileDataReadResultIterator,
    check_keys: bool,
) -> KernelResult<FileDataReadResultIterator> {
    let mut inline = false;
    let mut sidecars = Vec::new();
    let mut keys = input::FileKeys::default();
    for batch in batches {
        let data = ArrowEngineData::try_from_engine_data(batch?)?;
        let record = data.record_batch();
        if check_keys {
            keys.check_record(record)?;
        }
        inline |= ["add", "remove"].iter().any(|name| {
            let action = record.column_by_name(name).unwrap();
            action.null_count() < action.len()
        });
        let sidecar = record.column_by_name("sidecar").unwrap();
        if sidecar.null_count() < sidecar.len() {
            let field = record
                .schema()
                .fields()
                .iter()
                .find(|f| f.name() == "sidecar")
                .unwrap()
                .clone();
            let selected = if let Some(nulls) = sidecar.nulls() {
                filter(
                    sidecar.as_ref(),
                    &BooleanArray::new(nulls.inner().clone(), None),
                )?
            } else {
                sidecar.clone()
            };
            sidecars.push((field, selected));
        }
    }
    if inline && !sidecars.is_empty() {
        return Err(KernelError::invalid_checkpoint(
            "sidecar checkpoint has embedded file actions",
        ));
    }
    let mut listings = SidecarListings::default();
    let sidecars = sidecars
        .into_iter()
        .map(|(field, column)| {
            let sidecar = column.as_any().downcast_ref::<StructArray>().unwrap();
            let sidecar = fill_legacy_sidecar_metadata(io, sidecar, &mut listings)?;
            let paths = cast(
                sidecar.column_by_name("path").unwrap().as_ref(),
                &ArrowDataType::Utf8,
            )?;
            let paths = paths.as_any().downcast_ref::<StringArray>().unwrap();
            let base = Url::parse(ROOT)?.join("_delta_log/_sidecars/")?;
            let mut locations = io.sidecars.lock().unwrap();
            for path in paths.iter().flatten() {
                locations.insert(base.join(path)?);
            }
            let data = StructArray::try_new(vec![field].into(), vec![Arc::new(sidecar)], None)?;
            Ok(Box::new(ArrowEngineData::new(data.into())) as Box<dyn EngineData>)
        })
        .collect::<KernelResult<Vec<_>>>()?;
    Ok(Box::new(sidecars.into_iter().map(Ok)))
}

type SidecarListings = PlHashMap<String, PlHashMap<String, (u64, Option<i64>)>>;

fn fill_legacy_sidecar_metadata(
    io: &HostIo,
    sidecar: &StructArray,
    listings: &mut SidecarListings,
) -> KernelResult<StructArray> {
    let times = sidecar.column_by_name("modificationTime").unwrap();
    if times.null_count() == 0 {
        return Ok(sidecar.clone());
    }
    let times = times.as_any().downcast_ref::<Int64Array>().unwrap();
    let sizes = sidecar
        .column_by_name("sizeInBytes")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let paths = cast(
        sidecar.column_by_name("path").unwrap().as_ref(),
        &ArrowDataType::Utf8,
    )?;
    let paths = paths.as_any().downcast_ref::<StringArray>().unwrap();
    let mut new_times = Vec::with_capacity(sidecar.len());
    let mut new_sizes = Vec::with_capacity(sidecar.len());
    for ((path, time), size) in paths.iter().zip(times.iter()).zip(sizes.iter()) {
        if let (Some(path), None, Some(size)) = (path, time, size)
            && size >= 0
        {
            let (actual_size, actual_time) = legacy_sidecar_stat(io, path, listings)?;
            new_times.push(Some(actual_time));
            new_sizes.push(Some(actual_size));
        } else {
            new_times.push(time);
            new_sizes.push(size);
        }
    }
    let sidecar = replace_child(sidecar, "modificationTime", |_| {
        Ok(Arc::new(Int64Array::from(new_times)))
    })?;
    replace_child(&sidecar, "sizeInBytes", |_| {
        Ok(Arc::new(Int64Array::from(new_sizes)))
    })
}

fn legacy_sidecar_stat(
    io: &HostIo,
    path: &str,
    listings: &mut SidecarListings,
) -> KernelResult<(i64, i64)> {
    let base = Url::parse(ROOT)?.join("_delta_log/_sidecars/")?;
    let location = base.join(path)?;
    let missing = || {
        KernelError::invalid_checkpoint(format!(
            "sidecar {path} has no modificationTime and host metadata is unavailable"
        ))
    };
    if !location.as_str().starts_with(base.as_str()) {
        return Err(missing());
    }
    let url = io.metadata_path(&location)?;
    let (parent, _) = url.rsplit_once('/').ok_or_else(missing)?;
    let object = runtime::block_on(None, async {
        let mut live = io.live.lock().await;
        let live = live.as_mut().ok_or_else(closed).map_err(HostError::from)?;
        if !listings.contains_key(parent) {
            let objects = live
                .storage
                .list_with_sizes(&format!("{parent}/"), None)
                .await
                .map_err(HostError::from)?;
            listings.insert(
                parent.to_owned(),
                objects
                    .into_iter()
                    .map(|(object, size)| (object.url, (size, object.modified_ms)))
                    .collect(),
            );
        }
        let Some(&(listed_size, modified_ms)) =
            listings.get(parent).and_then(|listed| listed.get(&url))
        else {
            return Ok(None);
        };
        // Kernel uses this FileMeta for sidecar I/O only. A host may omit mtime too.
        // Pin to the actual listing size before reads.
        let size = live.pin(&url, Some(listed_size)).await?;
        Ok::<_, HostError>(Some((size, modified_ms.unwrap_or(MISSING_MODIFIED_MS))))
    })
    .map_err(HostError::into_kernel)?;
    let (size, time) = object.ok_or_else(missing)?;
    let size = i64::try_from(size).map_err(KernelError::invalid_checkpoint)?;
    Ok((size, time))
}

pub(super) struct HostParquet {
    pub(super) io: Arc<HostIo>,
    pub(super) inner: Arc<dyn ParquetHandler>,
    pub(super) bounds: Arc<CheckpointBounds>,
}

impl ParquetHandler for HostParquet {
    fn read_parquet_files(
        &self,
        files: &[FileMeta],
        schema: SchemaRef,
        predicate: Option<PredicateRef>,
    ) -> KernelResult<FileDataReadResultIterator> {
        let discover_sidecars = schema.fields().len() == 1 && schema.field("sidecar").is_some();
        let schema = with_checkpoint_stats_text(snapshot_projection(schema)?)?;
        let schema = host_sidecar_projection(schema, discover_sidecars, false)?;
        if discover_sidecars {
            let aliases = files
                .iter()
                .map(|file| self.io.alias_parquet(file))
                .collect::<KernelResult<Vec<_>>>()?;
            let batches = self.inner.read_parquet_files(&aliases, schema, predicate)?;
            return discover_checkpoint_sidecars(&self.io, batches, false);
        }
        if self.io.schema_only
            && schema.num_fields() == 2
            && schema.field("protocol").is_some()
            && schema.field("metaData").is_some()
        {
            let aliases = files
                .iter()
                .map(|file| self.io.alias_parquet(file))
                .collect::<KernelResult<Vec<_>>>()?;
            let batches = self.inner.read_parquet_files(&aliases, schema, predicate)?;
            return input::check_checkpoint_metadata(batches);
        }
        let inspect_bounds = CheckpointBounds::wanted(&schema);
        let check_rows = !self.io.schema_only && schema.field("add").is_some();
        let check_metadata_error = !self.io.schema_only && schema.field("metaData").is_some();
        let original_columns = (0..schema.num_fields()).collect::<Arc<[_]>>();
        let schema = if check_rows {
            input::checkpoint_projection(&schema)?
        } else {
            schema
        };
        let inspect_partitions = partition_map(&schema).is_some();
        let aliases = files
            .iter()
            .map(|file| Ok((file.location.clone(), self.io.alias_parquet(file)?)))
            .collect::<KernelResult<Vec<_>>>()?;
        if !inspect_bounds && !inspect_partitions && !check_rows && !check_metadata_error {
            let aliases: Vec<_> = aliases.into_iter().map(|(_, alias)| alias).collect();
            return self.inner.read_parquet_files(&aliases, schema, predicate);
        }

        let bounds = self.bounds.clone();
        let inner = self.inner.clone();
        let io = self.io.clone();
        Ok(Box::new(aliases.into_iter().flat_map(
            move |(location, alias)| {
                let read = || {
                    let is_sidecar = check_rows && io.sidecars.lock().unwrap().contains(&location);
                    let layout = (inspect_bounds || inspect_partitions)
                        .then(|| bounds.layout(&location, &alias))
                        .transpose()?;
                    let binary_partitions = layout.as_ref().filter(|layout| {
                        inspect_partitions
                            && (layout.binary_partition_key || layout.binary_partition_value)
                    });
                    let read_schema = match binary_partitions {
                        Some(layout) => binary_partition_projection(&schema, layout)?,
                        None => schema.clone(),
                    };
                    let binary_partitions = binary_partitions.is_some();
                    // Kernel's Add-presence predicate skips entire row groups with no Adds.
                    // For sidecars a non-file action in one of those groups is still invalid.
                    let predicate = if is_sidecar { None } else { predicate.clone() };
                    let batches = inner.read_parquet_files(
                        std::slice::from_ref(&alias),
                        read_schema,
                        predicate,
                    )?;
                    let original_columns = original_columns.clone();
                    let inner = inner.clone();
                    Ok(Box::new(batches.map(move |batch| {
                        let mut batch = match batch {
                            Err(error) if check_metadata_error => {
                                return Err(input::checkpoint_error(error, || {
                                    let schema = Arc::new(StructType::try_new([])?);
                                    let schema = input::checkpoint_projection(&schema)?;
                                    inner.read_parquet_files(
                                        std::slice::from_ref(&alias),
                                        schema,
                                        None,
                                    )
                                }));
                            },
                            batch => batch?,
                        };
                        if check_rows {
                            batch = input::check_checkpoint(batch, &original_columns, is_sidecar)?;
                        }
                        if let Some(layout) = &layout
                            && inspect_bounds
                            && !layout.unsafe_bounds.is_empty()
                        {
                            batch = CheckpointBounds::suppress(batch, &layout.unsafe_bounds)?;
                        }
                        if binary_partitions {
                            batch = restore_partition_text(batch)?;
                        }
                        Ok(batch)
                    })) as FileDataReadResultIterator)
                };
                read().unwrap_or_else(|error| Box::new(std::iter::once(Err(error))))
            },
        )))
    }

    fn read_parquet_footer(&self, file: &FileMeta) -> KernelResult<ParquetFooter> {
        self.bounds
            .footer(&file.location, &self.io.alias_parquet(file)?)
    }

    fn write_parquet_file(
        &self,
        _: Url,
        _: KernelResultIteratorStatic<Box<dyn EngineData>>,
    ) -> KernelResult<FileSize> {
        Err(read_only())
    }
}

fn partition_map(schema: &StructType) -> Option<&delta_kernel::schema::MapType> {
    let DataType::Struct(add) = schema.field("add")?.data_type() else {
        return None;
    };
    let DataType::Map(map) = add.field("partitionValues")?.data_type() else {
        return None;
    };
    (map.key_type == DataType::STRING && map.value_type == DataType::STRING).then_some(map)
}

fn binary_partition_projection(
    schema: &StructType,
    layout: &CheckpointLayout,
) -> KernelResult<SchemaRef> {
    let mut fields: Vec<_> = schema.fields().cloned().collect();
    let add = fields
        .iter_mut()
        .find(|field| field.name() == "add")
        .unwrap();
    let DataType::Struct(children) = add.data_type() else {
        unreachable!("partition map projection requires add struct");
    };
    let mut children: Vec<_> = children.fields().cloned().collect();
    let partitions = children
        .iter_mut()
        .find(|field| field.name() == "partitionValues")
        .unwrap();
    let mut map = partition_map(schema).unwrap().clone();
    if layout.binary_partition_key {
        map.key_type = DataType::BINARY;
    }
    if layout.binary_partition_value {
        map.value_type = DataType::BINARY;
    }
    partitions.data_type = map.into();
    add.data_type = StructType::try_new(children)?.into();
    Ok(Arc::new(StructType::try_new(fields)?))
}

// partitionValues is a text map in the Delta log. Read unannotated BYTE_ARRAY
// keys and values in their physical type, then strictly decode them as UTF-8.
fn restore_partition_text(batch: Box<dyn EngineData>) -> KernelResult<Box<dyn EngineData>> {
    let batch = ArrowEngineData::try_from_engine_data(batch)?;
    let batch: RecordBatch = (*batch).into();
    let batch = replace_child(&StructArray::from(batch), "add", |add| {
        let add = add.as_any().downcast_ref::<StructArray>().unwrap();
        replace_child(add, "partitionValues", |map| {
            let ArrowDataType::Map(entries, sorted) = map.data_type() else {
                unreachable!("partitionValues projection requires map");
            };
            let ArrowDataType::Struct(fields) = entries.data_type() else {
                unreachable!("Arrow map entries have struct datatype");
            };
            let mut fields = fields.to_vec();
            // Kernel's MAP_TO_STRUCT evaluator reads both entries as StringArray.
            for field in &mut fields {
                *field = Arc::new(field.as_ref().clone().with_data_type(ArrowDataType::Utf8));
            }
            let entries = entries
                .as_ref()
                .clone()
                .with_data_type(ArrowDataType::Struct(fields.into()));
            cast_with_options(
                map.as_ref(),
                &ArrowDataType::Map(Arc::new(entries), *sorted),
                &CastOptions {
                    safe: false,
                    ..Default::default()
                },
            )
            .map_err(|error| {
                KernelError::invalid_checkpoint(format!(
                    "checkpoint partitionValues is not valid UTF-8: {error}"
                ))
            })
        })
        .map(|array| Arc::new(array) as ArrayRef)
    })?;
    Ok(Box::new(ArrowEngineData::new(batch.into())))
}

fn replace_child(
    parent: &StructArray,
    name: &str,
    replace: impl FnOnce(&ArrayRef) -> KernelResult<ArrayRef>,
) -> KernelResult<StructArray> {
    let mut fields = parent.fields().to_vec();
    let mut columns = parent.columns().to_vec();
    let index = fields
        .iter()
        .position(|field| field.name() == name)
        .unwrap();
    let column = replace(&columns[index])?;
    fields[index] = Arc::new(
        fields[index]
            .as_ref()
            .clone()
            .with_data_type(column.data_type().clone()),
    );
    columns[index] = column;
    Ok(StructArray::try_new_with_length(
        fields.into(),
        columns,
        parent.nulls().cloned(),
        parent.len(),
    )?)
}

/// Kernel 0.30.1 projects the full Add and Remove types into snapshot replay. The
/// Polars exporter does not use the add mtime, and neither reconciliation nor the
/// exporter uses dataChange. Keep missing values null and retain all other checks.
/// This projection is for snapshot planning, where we consume scan_metadata batches
/// directly. Kernel's ScanFile visitor and change-data reads require these fields.
fn snapshot_projection(schema: SchemaRef) -> KernelResult<SchemaRef> {
    let mut fields = schema.fields().cloned().collect::<Vec<_>>();
    let mut changed = false;
    for action in &mut fields {
        let name = action.name().as_str();
        if !matches!(name, "add" | "remove") {
            continue;
        }
        let DataType::Struct(children) = action.data_type() else {
            continue;
        };
        let mut children = children.fields().cloned().collect::<Vec<_>>();
        let mut changed_action = false;
        for field in &mut children {
            let optional = (field.name() == "dataChange" && field.data_type == DataType::BOOLEAN)
                || (name == "add"
                    && field.name() == "modificationTime"
                    && field.data_type == DataType::LONG);
            if optional && !field.nullable {
                field.nullable = true;
                changed_action = true;
            }
        }
        if changed_action {
            action.data_type = StructType::try_new(children)?.into();
            changed = true;
        }
    }
    if changed {
        Ok(Arc::new(StructType::try_new(fields)?))
    } else {
        Ok(schema)
    }
}
