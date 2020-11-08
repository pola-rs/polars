//! This program is an example of an aggregator that computes the mean of a
//! dataframe in chunks. That is:
//! - It will aggregate N chunks of M files.
//! - The M files in each chunk will be loaded in parallel and aggregated
//!   into a partial aggregated DataFrame. This is done in order to
//!   reduce the amount of data loaded into memory. It will generate
//!   N partial aggregated dataframe, the same as chunks there are.
//! - The N partial aggregated dataframes will be aggregated into
//!   a global aggregated dataframe, that will contain the aggregation
//!   of the data of all the files to aggregate.
//! - Compute the mean over the global aggregated dataframe. NOTE: It
//!   is important to notice that it is not possible to use the `mean`
//!   function when using partial aggregations as we need the global
//!   information of the total sum by category and the toal count by
//!   category.
//!
//! # Use Case
//!
//! This program is useful in cases when dealing with big dataframes
//! that do not fit in the memory. In these cases, loading the whole
//! dataframe into memory is not possible, but loading a partial view
//! of the dataframe and performing aggregations by a field with low
//! cardinality is possible. Because after every aggregation the
//! amount of data in memory will be reduced.
//!
//! By limiting the load of files to M, we are not loading the whole
//! data into memory, but just a subset. After the aggregation, the
//! data will be reduced and then, and only then, the data next chunk
//! will be load.
//!
//! # Output
//!
//! Partial Aggregated DataFrame:
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//! | category     | calories_sum | calories_count | fats_g_sum | fats_g_count | sugars_g_sum | sugars_g_count |
//! | ---          | ---          | ---            | ---        | ---          | ---          | ---            |
//! | str          | u64          | u32            | f64        | u32          | f64          | u32            |
//! +==============+==============+================+============+==============+==============+================+
//! | "meat"       | 1080         | 10             | 73         | 10           | 3            | 10             |
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//! | "vegetables" | 386          | 14             | 2.8        | 14           | 44           | 14             |
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//! | "fruit"      | 811          | 14             | 9.9        | 14           | 130          | 14             |
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//! | "seafood"    | 2491         | 16             | 91.1       | 16           | 15           | 16             |
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//!
//! Partial Aggregated DataFrame:
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//! | category     | calories_sum | calories_count | fats_g_sum | fats_g_count | sugars_g_sum | sugars_g_count |
//! | ---          | ---          | ---            | ---        | ---          | ---          | ---            |
//! | str          | u64          | u32            | f64        | u32          | f64          | u32            |
//! +==============+==============+================+============+==============+==============+================+
//! | "meat"       | 1112         | 10             | 77         | 10           | 11           | 10             |
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//! | "vegetables" | 396          | 14             | 3.1        | 14           | 51           | 14             |
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//! | "fruit"      | 898          | 14             | 18         | 14           | 155          | 14             |
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//! | "seafood"    | 2547         | 16             | 98.1       | 16           | 24           | 16             |
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//!
//! Partial Aggregated DataFrame:
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//! | category     | calories_sum | calories_count | fats_g_sum | fats_g_count | sugars_g_sum | sugars_g_count |
//! | ---          | ---          | ---            | ---        | ---          | ---          | ---            |
//! | str          | u64          | u32            | f64        | u32          | f64          | u32            |
//! +==============+==============+================+============+==============+==============+================+
//! | "fruit"      | 409          | 7              | 4.2        | 7            | 61           | 7              |
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//! | "vegetables" | 191          | 7              | 0.4        | 7            | 16           | 7              |
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//! | "seafood"    | 1234         | 8              | 42.3       | 8            | 11           | 8              |
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//! | "meat"       | 453          | 5              | 27         | 5            | 3            | 5              |
//! +--------------+--------------+----------------+------------+--------------+--------------+----------------+
//!
//! Final Aggregated DataFrame:
//! +--------------+------------------+--------------------+----------------+------------------+------------------+--------------------+
//! | category     | calories_sum_sum | calories_count_sum | fats_g_sum_sum | fats_g_count_sum | sugars_g_sum_sum | sugars_g_count_sum |
//! | ---          | ---              | ---                | ---            | ---              | ---              | ---                |
//! | str          | u64              | u32                | f64            | u32              | f64              | u32                |
//! +==============+==================+====================+================+==================+==================+====================+
//! | "meat"       | 2645             | 25                 | 177            | 25               | 17               | 25                 |
//! +--------------+------------------+--------------------+----------------+------------------+------------------+--------------------+
//! | "vegetables" | 973              | 35                 | 6.3            | 35               | 111              | 35                 |
//! +--------------+------------------+--------------------+----------------+------------------+------------------+--------------------+
//! | "fruit"      | 2118             | 35                 | 32.1           | 35               | 346              | 35                 |
//! +--------------+------------------+--------------------+----------------+------------------+------------------+--------------------+
//! | "seafood"    | 6272             | 40                 | 231.5          | 40               | 50               | 40                 |
//! +--------------+------------------+--------------------+----------------+------------------+------------------+--------------------+
//!
//! Final Aggregated DataFrame with Mean Columns:
//! +--------------+---------------+-------------+---------------+
//! | category     | calories_mean | fats_g_mean | sugars_g_mean |
//! | ---          | ---           | ---         | ---           |
//! | str          | f64           | f64         | f64           |
//! +==============+===============+=============+===============+
//! | "meat"       | 105.8         | 7.08        | 0.68          |
//! +--------------+---------------+-------------+---------------+
//! | "vegetables" | 27.8          | 0.18        | 3.171         |
//! +--------------+---------------+-------------+---------------+
//! | "fruit"      | 60.514        | 0.917       | 9.886         |
//! +--------------+---------------+-------------+---------------+
//! | "seafood"    | 156.8         | 5.788       | 1.25          |
//! +--------------+---------------+-------------+---------------+
use polars::prelude::{
    ArrowDataType, CsvReader, DataFrame, Field, Result as PolarResult, Schema, SerReader, Series,
};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::error::Error;
use std::fs::{self, File};
use std::io::Result as IoResult;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// The number of files which will be loaded in parallel.
const FILES_IN_PARALLEL: usize = 2;

// Return a Vector which contains the path to all the files contained
// in the directory provided as input.
fn read_dir<P: AsRef<Path>>(directory: P) -> IoResult<Vec<PathBuf>> {
    fs::read_dir(directory)?
        .map(|res_entry| res_entry.map(|entry| entry.path()))
        .collect()
}

// Return the schema of the DataFrame read from the CSV.
fn get_schema() -> Schema {
    Schema::new(vec![
        Field::new("category", ArrowDataType::Utf8, false),
        Field::new("calories", ArrowDataType::UInt64, false),
        Field::new("fats_g", ArrowDataType::Float64, false),
        Field::new("sugars_g", ArrowDataType::Float64, false),
    ])
}

// Read the input CSV `path` as DataFrame using the schema
// provided for `get_schema`.
fn read_csv<P: AsRef<Path>>(path: P) -> PolarResult<DataFrame> {
    let schema = get_schema();

    let file = File::open(path).expect("Cannot open file.");

    CsvReader::new(file)
        .with_schema(Arc::new(schema))
        .has_header(true)
        .finish()
}

// Compute the mean of a field by using:
// mean_column = sum_column / count_column.
//
// # Input
//
// dataframe: The dataframe where to get the sum_column and count_column.
//     These columns will be droped and extracted from the dataframe, so the
//     dataframe shall be mutable.
// sum_column_name: The name of the sum column in the `dataframe`.
// count_column_name: The name of the count column in the `dataframe`.
// mean_column_name: The name of the mean serie to be returned.
fn compute_mean(
    dataframe: &mut DataFrame,
    sum_column_name: &str,
    count_column_name: &str,
    mean_column_name: &str,
) -> PolarResult<Series> {
    // Get the sum column from dataframe as float.
    let sum_column = dataframe
        .drop_in_place(sum_column_name)?
        .cast_with_arrow_datatype(&ArrowDataType::Float64)?;

    // Get the count column from dataframe as float.
    let count_column = dataframe
        .drop_in_place(count_column_name)?
        .cast_with_arrow_datatype(&ArrowDataType::Float64)?;

    // Compute the mean serie and rename to the `mean_column_name` provided
    // as input.
    let mut mean_column = &sum_column / &count_column;
    mean_column.rename(mean_column_name);

    // Return successfully the serie.
    Ok(mean_column)
}

// Helper function for fold DataFrames. It appends DataFrames to the accumulator,
// if the acumulator is the default DataFrame, then, return the right DataFrame, as the
// accumulator.
fn right_or_append(mut accumulator: DataFrame, right: DataFrame) -> PolarResult<DataFrame> {
    if accumulator.width() == 0 {
        Ok(right)
    } else {
        accumulator.vstack(&right)?;
        Ok(accumulator)
    }
}

// This function reads in parallel the files in the `paths` slice, and
// returns the aggregation of all the files in the slice.
//
// The steps are:
// 1. Read DataFrame from CSV in parallel.
// 2. Append the files to the same DataFrame as soon as the DataFrame
//    is available.
// 3. Group by category.
// 4. Aggregate computing the sum and the count of calories, fats_g and
//    sugars_g. At this point the schema will change to: ['category',
//    'calories_sum', 'calories_count', 'fats_g_sum', 'fats_g_count',
//    'sugars_g_sum', 'sugars_g_count']
//
// The input is a slice of paths to CSV files.
// The output is the aggregated DataFrame for all CSVs in the slice.
fn process_files_parallel(paths: &[PathBuf]) -> PolarResult<DataFrame> {
    paths
        .into_par_iter()
        .map(read_csv)
        .try_reduce(DataFrame::default, right_or_append)?
        .groupby(&["category"])?
        .agg(&[
            ("calories", &["sum", "count"]),
            ("fats_g", &["sum", "count"]),
            ("sugars_g", &["sum", "count"]),
        ])
}

// Compute the mean for every field:
// - calories_mean from calories_sum_sum and calories_count_sum
// - fats_g_mean from fats_g_sum_sum and fats_g_count_sum
// - sugars_g_mean from sugars_g_sum_sum and sugars_g_count_sum
//
// The input is the dataframe used to get the '${field}_count_sum' and
// '${field}_sum_sum' fiels. It shall be mutable, as the fields are going
// to be dropped when computed the '${field}_mean'.
//
// The output is a result containg the Vector of mean Series computed.
fn compute_all_means(dataframe: &mut DataFrame) -> PolarResult<Vec<Series>> {
    const SERIES_NAMES: &[(&str, &str, &str)] = &[
        ("calories_sum_sum", "calories_count_sum", "calories_mean"),
        ("fats_g_sum_sum", "fats_g_count_sum", "fats_g_mean"),
        ("sugars_g_sum_sum", "sugars_g_count_sum", "sugars_g_mean"),
    ];

    let mut result = Vec::with_capacity(SERIES_NAMES.len());
    for (sum_column_name, count_column_name, mean_column_name) in SERIES_NAMES {
        let mean_column = compute_mean(
            dataframe,
            sum_column_name,
            count_column_name,
            mean_column_name,
        )?;
        result.push(mean_column);
    }

    Ok(result)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Get the directory where dataset are located.
    let dataset_dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), "datasets"].iter().collect();

    // Read all the files in the directory and sort them.
    //
    // Notice that the sorting is just for debugging reasons, to keep the order
    // of the partial aggregated DataFrames stable. In a production system it
    // is not necessary to perform any sorting.
    let mut paths = read_dir(dataset_dir)?;
    paths.sort_unstable();

    // Process files in parallel in chunks of size `FILES_IN_PARALLEL`.
    //
    // Basically it will perform the following steps:
    // 1. Read in parallel up to `FILES_IN_PARALLEL`.
    // 2. Aggregate in parallel `FILES_IN_PARALLEL` getting a partial
    //    aggregation of the `FILES_IN_PARALLEL`. At this point the
    //    schema will change to ['category', 'calories_sum', 'calories_count',
    //    'fats_g_sum', 'fats_g_count', 'sugars_g_sum', 'sugars_g_count'].
    // 3. Append sequencially each partial dataframe to the final dataframe.
    // 4. Group by category.
    // 5. Aggregate summing the sums and the counts of the partial aggregation
    //    to get the global sums '${field}_sum_sum' and the global counts
    //    '${field}_count_sum. This fields are needed to compute the global
    //    mean. Again, the schema of the dataframe will change.
    let mut main_df = paths
        .chunks(FILES_IN_PARALLEL)
        .try_fold(DataFrame::default(), |main_df, paths| {
            let df = process_files_parallel(paths)?;

            println!("Partial Aggregated DataFrame:\n{}", df);

            right_or_append(main_df, df)
        })?
        .groupby(&["category"])?
        .select(&[
            "calories_sum",
            "calories_count",
            "fats_g_sum",
            "fats_g_count",
            "sugars_g_sum",
            "sugars_g_count",
        ])
        .sum()?;

    println!("Final Aggregated DataFrame:\n{}", main_df);

    // Compute the mean for every field:
    // - calories_mean from calories_sum_sum and calories_count_sum
    // - fats_g_mean from fats_g_sum_sum and fats_g_count_sum
    // - sugars_g_mean from sugars_g_sum_sum and sugars_g_count_sum
    // The ${field}_sum_sum and ${field}_count_sum colums will be
    // droped after computing the mean as they are not needed anymore.
    let mean_series = compute_all_means(&mut main_df)?;

    // Add the computed mean series to the main dataframe.
    // The schema at this point is ['category', 'calories_mean', 'fats_g_mean',
    // 'sugars_g_mean']
    mean_series
        .into_iter()
        .try_for_each(|serie| -> PolarResult<()> {
            main_df.add_column(serie)?;
            Ok(())
        })?;

    println!("Final Aggregated DataFrame with Mean Columns:\n{}", main_df);

    // Return Successfully.
    Ok(())
}
