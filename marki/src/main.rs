use polars::prelude::*;
use polars::lazy::{dsl::StrpTimeOptions};
use polars::time::Duration as PolarsDuration;

use polars::df;
use chrono::{NaiveDate, Duration};

fn main() {
    let start_date = NaiveDate::from_ymd_opt(2001, 1, 1).expect("");
    let end_date = NaiveDate::from_ymd_opt(2001, 1, 4).expect("");

    let duration = Duration::days(1);

    let date_seq = date_sequence(start_date, end_date, duration, ClosedWindow::Both).unwrap();
    let date_vec = tile(&date_seq, 2 as usize).unwrap();

    let group_vec = repeat(&vec!["a", "b"], 4 as usize).unwrap();

    let name_vec = repeat(&vec!["katrien", "marc"], 4 as usize).unwrap();

    let values = vec![2, 3, 9, 1, 3, 2, 9, 10];

    let pl_frame = df!(
        "date" => date_vec, 
        "group" => group_vec,
        "name" => name_vec,
        "value" => values, 
    ).unwrap();

    //filter one observation out of the dataframe (for upsampling)
    let date_format = String::from("%Y-%m-%d");
    let option: StrpTimeOptions = StrpTimeOptions { 
        date_dtype: DataType::Date, fmt: Some(date_format), ..Default::default()
    };
    let date_value = lit("2001-01-03").str().strptime(option);

    let mask = (col("group").eq(lit("a")).and(col("date").eq(date_value))).not();
    let pl_frame = pl_frame.lazy()
                .filter(mask).collect().unwrap();
                    
    // Set-up upsampling
    let every_day = PolarsDuration::parse("1d");
    let no_offset = PolarsDuration::parse("0d");

    let group_columns = vec!["group", "name"];

    let pl_frame = pl_frame.upsample(&group_columns, "date", every_day, no_offset).unwrap();

    // for column in group_columns {
    //     let new_group = pl_frame.column(column).unwrap().fill_null(FillNullStrategy::Forward(None)).unwrap();
    //     pl_frame.with_column(new_group).unwrap();
    // }

    println!("{:#?}", pl_frame);
}

fn repeat<T: Copy>(vector: &Vec<T>, n: usize) -> PolarsResult<Vec<T>>{
    let container = vector.into_iter().flat_map(|v| std::iter::repeat(*v).take(n)).collect();

    Ok(container)
}

fn tile<T: Clone>(vector: &Vec<T>, n: usize) -> PolarsResult<Vec<T>>{
    let mut container = Vec::with_capacity(n * vector.len());
    for _ in 0..n {
        container.extend_from_slice(vector);
    }

    Ok(container)
}

// Should think about getting this as a feature too. 
fn date_sequence(start_date: NaiveDate, end_date: NaiveDate, interval: Duration, closed: ClosedWindow) -> PolarsResult<Vec<NaiveDate>> {
        let mut current_date = start_date.clone();
        let mut date_vec = Vec::new();

        match closed {
            ClosedWindow::Both => {
                while current_date <= end_date {
                    date_vec.push(current_date);
                    current_date += interval;
                }
            }
            ClosedWindow::Left => {
                while current_date < end_date {
                    date_vec.push(current_date);
                    current_date += interval;
                }
            }
            ClosedWindow::Right => {
                current_date = end_date.clone();
                while current_date > start_date {
                    date_vec.push(current_date);
                    current_date -= interval;
                }
                date_vec.reverse();
            }
            ClosedWindow::None => {
                current_date += interval;
                while current_date < end_date {
                    date_vec.push(current_date);
                    current_date += interval;
                }
            }
        }

        Ok(date_vec)
}