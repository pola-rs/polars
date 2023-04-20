use polars::prelude::*;
use chrono::{NaiveDate, Duration};

fn main() {
    let start_date = NaiveDate::from_ymd_opt(2001, 1, 1).expect("");
    let end_date = NaiveDate::from_ymd_opt(2001, 1, 4).expect("");

    let duration = Duration::days(1);

    let date_seq = date_sequence(start_date, end_date, duration, ClosedWindow::Both).unwrap();
    let date_vec = tile(&date_seq, 2 as usize).unwrap();

    let group_vec = repeat(&vec!["a", "b"], 4 as usize).unwrap();

    let values = vec![2, 3, 9, 1, 3, 2, 9, 10];

    let pl_frame = df!(
        "date" => date_vec, 
        "group" => group_vec,
        "value" => values, 
    ).unwrap();

    //filter one observation out of the dataframe (for upsampling)
    

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