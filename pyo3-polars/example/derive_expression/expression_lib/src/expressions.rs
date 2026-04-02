use std::fmt::Write;

use polars::prelude::*;
use polars_plan::prelude::FieldsMapper;
use pyo3_polars::derive::{polars_expr, CallerContext};
use pyo3_polars::export::polars_core::POOL;
use serde::Deserialize;

#[derive(Deserialize)]
struct PigLatinKwargs {
    capitalize: bool,
}

fn pig_latin_str(value: &str, capitalize: bool, output: &mut String) {
    if let Some(first_char) = value.chars().next() {
        if capitalize {
            for c in value.chars().skip(1).map(|char| char.to_uppercase()) {
                write!(output, "{c}").unwrap()
            }
            write!(output, "AY").unwrap()
        } else {
            let offset = first_char.len_utf8();
            write!(output, "{}{}ay", &value[offset..], first_char).unwrap()
        }
    }
}

#[polars_expr(output_type=String)]
fn pig_latinnify(inputs: &[Series], kwargs: PigLatinKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(|value, output| {
        pig_latin_str(value, kwargs.capitalize, output)
    });
    Ok(out.into_series())
}

fn split_offsets(len: usize, n: usize) -> Vec<(usize, usize)> {
    if n == 1 {
        vec![(0, len)]
    } else {
        let chunk_size = len / n;

        (0..n)
            .map(|partition| {
                let offset = partition * chunk_size;
                let len = if partition == (n - 1) {
                    len - offset
                } else {
                    chunk_size
                };
                (partition * chunk_size, len)
            })
            .collect()
    }
}

/// This expression will run in parallel if the `context` allows it.
#[polars_expr(output_type=String)]
fn pig_latinnify_with_parallelism(
    inputs: &[Series],
    context: CallerContext,
    kwargs: PigLatinKwargs,
) -> PolarsResult<Series> {
    use rayon::prelude::*;
    let ca = inputs[0].str()?;

    if context.parallel() {
        let out: StringChunked = ca.apply_into_string_amortized(|value, output| {
            pig_latin_str(value, kwargs.capitalize, output)
        });
        Ok(out.into_series())
    } else {
        POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(ca.len(), n_threads);

            let chunks: Vec<_> = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let sliced = ca.slice(offset as i64, len);
                    let out = sliced.apply_into_string_amortized(|value, output| {
                        pig_latin_str(value, kwargs.capitalize, output)
                    });
                    out.downcast_iter().cloned().collect::<Vec<_>>()
                })
                .collect();

            Ok(
                StringChunked::from_chunk_iter(ca.name().clone(), chunks.into_iter().flatten())
                    .into_series(),
            )
        })
    }
}

#[polars_expr(output_type=Float64)]
fn jaccard_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].list()?;
    let b = inputs[1].list()?;
    crate::distances::naive_jaccard_sim(a, b).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn hamming_distance(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].str()?;
    let b = inputs[1].str()?;
    let out: UInt32Chunked =
        arity::binary_elementwise_values(a, b, crate::distances::naive_hamming_dist);
    Ok(out.into_series())
}

fn haversine_output(input_fields: &[Field]) -> PolarsResult<Field> {
    FieldsMapper::new(input_fields).map_to_float_dtype()
}

#[polars_expr(output_type_func=haversine_output)]
fn haversine(inputs: &[Series]) -> PolarsResult<Series> {
    let out = match inputs[0].dtype() {
        DataType::Float32 => {
            let start_lat = inputs[0].f32().unwrap();
            let start_long = inputs[1].f32().unwrap();
            let end_lat = inputs[2].f32().unwrap();
            let end_long = inputs[3].f32().unwrap();
            crate::distances::naive_haversine(start_lat, start_long, end_lat, end_long)?
                .into_series()
        },
        DataType::Float64 => {
            let start_lat = inputs[0].f64().unwrap();
            let start_long = inputs[1].f64().unwrap();
            let end_lat = inputs[2].f64().unwrap();
            let end_long = inputs[3].f64().unwrap();
            crate::distances::naive_haversine(start_lat, start_long, end_lat, end_long)?
                .into_series()
        },
        _ => unimplemented!(),
    };
    Ok(out)
}

/// The `DefaultKwargs` isn't very ergonomic as it doesn't validate any schema.
/// Provide your own kwargs struct with the proper schema and accept that type
/// in your plugin expression.
#[derive(Deserialize)]
pub struct MyKwargs {
    float_arg: f64,
    integer_arg: i64,
    string_arg: String,
    boolean_arg: bool,
}

/// If you want to accept `kwargs`. You define a `kwargs` argument
/// on the second position in you plugin. You can provide any custom struct that is deserializable
/// with the pickle protocol (on the rust side).
#[polars_expr(output_type=String)]
fn append_kwargs(input: &[Series], kwargs: MyKwargs) -> PolarsResult<Series> {
    let input = &input[0];
    let input = input.cast(&DataType::String)?;
    let ca = input.str().unwrap();

    Ok(ca
        .apply_into_string_amortized(|val, buf| {
            write!(
                buf,
                "{}-{}-{}-{}-{}",
                val, kwargs.float_arg, kwargs.integer_arg, kwargs.string_arg, kwargs.boolean_arg
            )
            .unwrap()
        })
        .into_series())
}

#[polars_expr(output_type=Boolean)]
fn is_leap_year(input: &[Series]) -> PolarsResult<Series> {
    let input = &input[0];
    let ca = input.date()?;

    let out: BooleanChunked = ca
        .as_date_iter()
        .map(|opt_dt| opt_dt.map(|dt| dt.leap_year()))
        .collect_ca(ca.name().clone());

    Ok(out.into_series())
}

#[polars_expr(output_type=Boolean)]
fn panic(_input: &[Series]) -> PolarsResult<Series> {
    todo!()
}

#[derive(Deserialize)]
struct TimeZone {
    tz: String,
}

fn convert_timezone(input_fields: &[Field], kwargs: TimeZone) -> PolarsResult<Field> {
    FieldsMapper::new(input_fields).try_map_dtype(|dtype| match dtype {
        DataType::Datetime(tu, _) => Ok(DataType::Datetime(
            *tu,
            datatypes::TimeZone::opt_try_new(Some(kwargs.tz))?,
        )),
        _ => polars_bail!(ComputeError: "expected datetime"),
    })
}

/// This expression is for demonstration purposes as we have a dedicated
/// `convert_time_zone` in Polars.
#[polars_expr(output_type_func_with_kwargs=convert_timezone)]
fn change_time_zone(input: &[Series], kwargs: TimeZone) -> PolarsResult<Series> {
    let input = &input[0];
    let ca = input.datetime()?;

    let mut out = ca.clone();

    let Some(timezone) = datatypes::TimeZone::opt_try_new(Some(kwargs.tz))? else {
        polars_bail!(ComputeError: "expected timezone")
    };

    out.set_time_zone(timezone)?;
    Ok(out.into_series())
}
