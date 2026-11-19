use polars_arrow::array::*;
use polars_arrow::datatypes::Field;
use polars_arrow::ffi;
use polars_error::{PolarsError, PolarsResult};

fn _test_round_trip(arrays: Vec<Box<dyn Array>>) -> PolarsResult<()> {
    let field = Field::new("a".into(), arrays[0].dtype().clone(), true);
    let iter = Box::new(arrays.clone().into_iter().map(Ok)) as _;

    let mut stream = Box::new(ffi::ArrowArrayStream::empty());

    *stream = ffi::export_iterator(iter, field.clone());

    // import
    let mut stream = unsafe { ffi::ArrowArrayStreamReader::try_new(stream)? };

    let mut produced_arrays: Vec<Box<dyn Array>> = vec![];
    while let Some(array) = unsafe { stream.next() } {
        produced_arrays.push(array?);
    }

    assert_eq!(produced_arrays, arrays);
    assert_eq!(stream.field(), &field);
    Ok(())
}

#[test]
fn round_trip() -> PolarsResult<()> {
    let array = Int32Array::from(&[Some(2), None, Some(1), None]);
    let array: Box<dyn Array> = Box::new(array);

    _test_round_trip(vec![array.clone(), array.clone(), array])
}

#[test]
fn stream_reader_try_new_invalid_argument_error_on_released_stream() {
    let released_stream = Box::new(ffi::ArrowArrayStream::empty());
    let reader = unsafe { ffi::ArrowArrayStreamReader::try_new(released_stream) };
    // poor man's assert_matches:
    match reader {
        Err(PolarsError::InvalidOperation(_)) => {},
        _ => panic!("ArrowArrayStreamReader::try_new did not return an InvalidArgumentError"),
    }
}

#[test]
fn panic_in_iterator_ends_stream() -> PolarsResult<()> {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let array: Box<dyn Array> = Box::new(Int32Array::from(&[Some(1), None]));
    let field = Field::new("a".into(), array.dtype().clone(), true);
    let calls = Arc::new(AtomicUsize::new(0));
    let iter = {
        let (array, calls) = (array.clone(), calls.clone());
        std::iter::from_fn(move || match calls.fetch_add(1, Ordering::Relaxed) {
            0 => Some(Ok(array.clone())),
            1 => panic!("iterator boom"),
            _ => Some(Ok(array.clone())),
        })
    };

    let stream = Box::new(ffi::export_iterator(Box::new(iter), field));
    let mut reader = unsafe { ffi::ArrowArrayStreamReader::try_new(stream)? };

    assert_eq!(unsafe { reader.next() }.unwrap()?, array);
    for _ in 0..2 {
        let err = unsafe { reader.next() }.unwrap().unwrap_err();
        assert!(err.to_string().contains("iterator boom"), "{err}");
    }
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    Ok(())
}
