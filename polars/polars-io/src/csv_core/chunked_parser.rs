use crate::csv::CsvEncoding;
use crate::csv_core::buffer::PrimitiveParser;
use crate::csv_core::utils::parse_bytes_with_encoding;
use crate::PhysicalIoExpr;
use crate::ScanAggregation;
use polars_core::prelude::*;

pub(crate) fn init_builders(
    projection: &[usize],
    capacity: usize,
    schema: &SchemaRef,
) -> Result<Vec<Builder>> {
    projection
        .iter()
        .map(|&i| field_to_builder(i, capacity, schema))
        .collect()
}

fn field_to_builder(i: usize, capacity: usize, schema: &SchemaRef) -> Result<Builder> {
    let field = schema.field(i).unwrap();
    let name = field.name();

    let builder = match field.data_type() {
        &DataType::Boolean => Builder::Boolean(BooleanChunkedBuilder::new(name, capacity)),
        &DataType::Int32 => Builder::Int32(PrimitiveChunkedBuilder::new(name, capacity)),
        &DataType::Int64 => Builder::Int64(PrimitiveChunkedBuilder::new(name, capacity)),
        &DataType::UInt32 => Builder::UInt32(PrimitiveChunkedBuilder::new(name, capacity)),
        #[cfg(feature = "dtype-u64")]
        &DataType::UInt64 => Builder::UInt64(PrimitiveChunkedBuilder::new(name, capacity)),
        &DataType::Float32 => Builder::Float32(PrimitiveChunkedBuilder::new(name, capacity)),
        &DataType::Float64 => Builder::Float64(PrimitiveChunkedBuilder::new(name, capacity)),
        &DataType::Utf8 => Builder::Utf8(Utf8ChunkedBuilder::new(name, capacity, capacity * 32)),
        other => {
            return Err(PolarsError::Other(
                format!("Unsupported data type {:?} when reading a csv", other).into(),
            ))
        }
    };
    Ok(builder)
}

fn builders_to_df(builders: Vec<Builder>) -> DataFrame {
    let columns = builders.into_iter().map(|b| b.into_series()).collect();
    DataFrame::new_no_checks(columns)
}

#[inline]
pub(crate) fn add_to_builders_core(
    builders: &mut [Builder],
    projection: &[usize],
    rows: &[PolarsCsvRecord],
    schema: &Schema,
    ignore_parser_error: bool,
    encoding: CsvEncoding,
) -> Result<()> {
    let dispatch = |(i, builder): (&usize, &mut Builder)| {
        let field = schema.field(*i).unwrap();
        match field.data_type() {
            DataType::Boolean => add_to_bool_core(rows, *i, builder.bool(), ignore_parser_error),
            DataType::Int32 => add_to_primitive_core(rows, *i, builder.i32(), ignore_parser_error),
            DataType::Int64 => add_to_primitive_core(rows, *i, builder.i64(), ignore_parser_error),
            DataType::UInt32 => add_to_primitive_core(rows, *i, builder.u32(), ignore_parser_error),
            #[cfg(feature = "dtype-u64")]
            DataType::UInt64 => add_to_primitive_core(rows, *i, builder.u64(), ignore_parser_error),
            DataType::Float32 => {
                add_to_primitive_core(rows, *i, builder.f32(), ignore_parser_error)
            }
            DataType::Float64 => {
                add_to_primitive_core(rows, *i, builder.f64(), ignore_parser_error)
            }
            DataType::Utf8 => add_to_utf8_builder_core(rows, *i, builder.utf8(), encoding),
            _ => panic!("datatype not supported"),
        }
    };

    projection.iter().zip(builders).try_for_each(dispatch)?;

    Ok(())
}

#[inline]
fn add_to_utf8_builder_core(
    rows: &[PolarsCsvRecord],
    col_idx: usize,
    builder: &mut Utf8ChunkedBuilder,
    encoding: CsvEncoding,
) -> Result<()> {
    for row in rows.iter() {
        let v = row.get(col_idx);
        match v {
            None => builder.append_null(),
            Some(bytes) => {
                if bytes.is_empty() {
                    builder.append_null()
                } else {
                    let s = parse_bytes_with_encoding(bytes, encoding)?;
                    builder.append_value(&s);
                }
            }
        }
    }
    Ok(())
}

#[inline]
fn add_to_primitive_core<T>(
    rows: &[PolarsCsvRecord],
    col_idx: usize,
    builder: &mut PrimitiveChunkedBuilder<T>,
    ignore_parser_errors: bool,
) -> Result<()>
where
    T: PolarsPrimitiveType + PrimitiveParser,
{
    // todo! keep track of line number for error reporting
    for (_row_index, row) in rows.iter().enumerate() {
        match row.get(col_idx) {
            Some(bytes) => {
                if bytes.is_empty() {
                    builder.append_null();
                    continue;
                }
                match T::parse(bytes) {
                    Ok(e) => builder.append_value(e),
                    Err(_) => {
                        if ignore_parser_errors {
                            builder.append_null();
                            continue;
                        }
                        return Err(PolarsError::Other(
                            format!(
                                "Error while parsing value {} for column {} as {:?}",
                                String::from_utf8_lossy(bytes),
                                col_idx,
                                T::get_dtype()
                            )
                            .into(),
                        ));
                    }
                }
            }
            None => builder.append_null(),
        }
    }
    Ok(())
}

#[inline]
fn add_to_bool_core(
    rows: &[PolarsCsvRecord],
    col_idx: usize,
    builder: &mut BooleanChunkedBuilder,
    ignore_parser_errors: bool,
) -> Result<()> {
    // todo! keep track of line number for error reporting
    for (_row_index, row) in rows.iter().enumerate() {
        match row.get(col_idx) {
            Some(bytes) => {
                if bytes.is_empty() {
                    builder.append_null();
                    continue;
                }
                if bytes.eq_ignore_ascii_case(b"false") {
                    builder.append_value(false);
                } else if bytes.eq_ignore_ascii_case(b"true") {
                    builder.append_value(true);
                } else if ignore_parser_errors {
                    builder.append_null();
                } else {
                    return Err(PolarsError::Other(
                        format!(
                            "Error while parsing value {} for column {} as {:?}",
                            String::from_utf8_lossy(bytes),
                            col_idx,
                            DataType::Boolean
                        )
                        .into(),
                    ));
                }
            }
            None => builder.append_null(),
        }
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) struct PolarsCsvRecord {
    out: Vec<u8>,
    ends: Vec<usize>,
    n_out: usize,
}

impl PolarsCsvRecord {
    #[inline]
    fn get(&self, index: usize) -> Option<&[u8]> {
        let start = match index.checked_sub(1).and_then(|idx| self.ends.get(idx)) {
            None => 0,
            Some(i) => *i,
        };
        let end = match self.ends.get(index) {
            Some(i) => *i,
            None => return None,
        };

        Some(&self.out[start..end])
    }

    #[inline]
    fn resize_out_buffer(&mut self) {
        let size = std::cmp::max(self.out.len() * 2, 128);
        self.out.resize(size, 0);
    }

    fn reset(&mut self) {
        self.ends.truncate(0);
        self.n_out = 0;
    }
}

impl Default for PolarsCsvRecord {
    fn default() -> Self {
        PolarsCsvRecord {
            out: vec![],
            ends: vec![],
            n_out: 0,
        }
    }
}

#[inline]
pub(crate) fn next_rows_core(
    rows: &mut Vec<PolarsCsvRecord>,
    mut bytes: &[u8],
    reader: &mut csv_core::Reader,
    batch_size: usize,
) -> (usize, usize) {
    let mut line_count = 0;
    let mut bytes_read = 0;
    loop {
        debug_assert!(rows.get(line_count).is_some());
        let mut record = unsafe { rows.get_unchecked_mut(line_count) };
        record.reset();

        use csv_core::ReadFieldResult;
        loop {
            let (result, n_in, n_out) = reader.read_field(bytes, &mut record.out[record.n_out..]);
            match result {
                ReadFieldResult::Field { record_end } => {
                    bytes_read += n_in;
                    bytes = &bytes[n_in..];
                    record.n_out += n_out;
                    record.ends.push(record.n_out);

                    if record_end {
                        line_count += 1;
                        break;
                    }
                }
                ReadFieldResult::OutputFull => {
                    record.resize_out_buffer();
                }
                ReadFieldResult::End | ReadFieldResult::InputEmpty => {
                    return (line_count, bytes_read);
                }
            }
        }
        if line_count == batch_size {
            break;
        }
    }
    (line_count, bytes_read)
}

pub(crate) fn finish_builder(
    builders: Vec<Builder>,
    parsed_dfs: &mut Vec<DataFrame>,
    predicate: Option<&Arc<dyn PhysicalIoExpr>>,
    aggregate: Option<&[ScanAggregation]>,
) -> Result<()> {
    let mut df = builders_to_df(builders);
    if let Some(predicate) = predicate {
        let s = predicate.evaluate(&df)?;
        let mask = s.bool().expect("filter predicates was not of type boolean");
        let local_df = df.filter(mask)?;
        if df.height() > 0 {
            df = local_df;
        }
    }
    // IMPORTANT the assumption of the aggregations is that all column are aggregated.
    // If that assumption is incorrect, aggregation should be None
    if let Some(aggregate) = aggregate {
        let cols = aggregate
            .iter()
            .map(|scan_agg| scan_agg.evaluate_batch(&df).unwrap())
            .collect();
        if cfg!(debug_assertions) {
            df = DataFrame::new(cols).unwrap();
        } else {
            df = DataFrame::new_no_checks(cols)
        }
    }
    parsed_dfs.push(df);
    Ok(())
}

pub(crate) enum Builder {
    Boolean(BooleanChunkedBuilder),
    Int32(PrimitiveChunkedBuilder<Int32Type>),
    Int64(PrimitiveChunkedBuilder<Int64Type>),
    #[cfg(feature = "dtype-u64")]
    UInt64(PrimitiveChunkedBuilder<UInt64Type>),
    UInt32(PrimitiveChunkedBuilder<UInt32Type>),
    Float32(PrimitiveChunkedBuilder<Float32Type>),
    Float64(PrimitiveChunkedBuilder<Float64Type>),
    Utf8(Utf8ChunkedBuilder),
}

impl Builder {
    fn bool(&mut self) -> &mut BooleanChunkedBuilder {
        match self {
            Builder::Boolean(builder) => builder,
            _ => panic!("implementation error"),
        }
    }
    fn i32(&mut self) -> &mut PrimitiveChunkedBuilder<Int32Type> {
        match self {
            Builder::Int32(builder) => builder,
            _ => panic!("implementation error"),
        }
    }
    fn i64(&mut self) -> &mut PrimitiveChunkedBuilder<Int64Type> {
        match self {
            Builder::Int64(builder) => builder,
            _ => panic!("implementation error"),
        }
    }
    fn u32(&mut self) -> &mut PrimitiveChunkedBuilder<UInt32Type> {
        match self {
            Builder::UInt32(builder) => builder,
            _ => panic!("implementation error"),
        }
    }
    #[cfg(feature = "dtype-u64")]
    fn u64(&mut self) -> &mut PrimitiveChunkedBuilder<UInt64Type> {
        match self {
            Builder::UInt64(builder) => builder,
            _ => panic!("implementation error"),
        }
    }
    fn f64(&mut self) -> &mut PrimitiveChunkedBuilder<Float64Type> {
        match self {
            Builder::Float64(builder) => builder,
            _ => panic!("implementation error"),
        }
    }
    fn f32(&mut self) -> &mut PrimitiveChunkedBuilder<Float32Type> {
        match self {
            Builder::Float32(builder) => builder,
            _ => panic!("implementation error"),
        }
    }
    fn utf8(&mut self) -> &mut Utf8ChunkedBuilder {
        match self {
            Builder::Utf8(builder) => builder,
            _ => panic!("implementation error"),
        }
    }

    fn into_series(self) -> Series {
        use Builder::*;
        match self {
            Utf8(b) => b.finish().into_series(),
            Int32(b) => b.finish().into_series(),
            Int64(b) => b.finish().into_series(),
            UInt32(b) => b.finish().into_series(),
            #[cfg(feature = "dtype-u64")]
            UInt64(b) => b.finish().into_series(),
            Float32(b) => b.finish().into_series(),
            Float64(b) => b.finish().into_series(),
            Boolean(b) => b.finish().into_series(),
        }
    }
}
