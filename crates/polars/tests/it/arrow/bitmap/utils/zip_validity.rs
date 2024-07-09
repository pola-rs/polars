use arrow::bitmap::utils::{BitmapIter, ZipValidity};
use arrow::bitmap::Bitmap;

#[test]
fn basic() {
    let a = Bitmap::from([true, false]);
    let a = Some(a.iter());
    let values = vec![0, 1];
    let zip = ZipValidity::new(values.into_iter(), a);

    let a = zip.collect::<Vec<_>>();
    assert_eq!(a, vec![Some(0), None]);
}

#[test]
fn complete() {
    let a = Bitmap::from([true, false, true, false, true, false, true, false]);
    let a = Some(a.iter());
    let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let zip = ZipValidity::new(values.into_iter(), a);

    let a = zip.collect::<Vec<_>>();
    assert_eq!(
        a,
        vec![Some(0), None, Some(2), None, Some(4), None, Some(6), None]
    );
}

#[test]
fn slices() {
    let a = Bitmap::from([true, false]);
    let a = Some(a.iter());
    let offsets = [0, 2, 3];
    let values = [1, 2, 3];
    let iter = offsets.windows(2).map(|x| {
        let start = x[0];
        let end = x[1];
        &values[start..end]
    });
    let zip = ZipValidity::new(iter, a);

    let a = zip.collect::<Vec<_>>();
    assert_eq!(a, vec![Some([1, 2].as_ref()), None]);
}

#[test]
fn byte() {
    let a = Bitmap::from([true, false, true, false, false, true, true, false, true]);
    let a = Some(a.iter());
    let values = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
    let zip = ZipValidity::new(values.into_iter(), a);

    let a = zip.collect::<Vec<_>>();
    assert_eq!(
        a,
        vec![
            Some(0),
            None,
            Some(2),
            None,
            None,
            Some(5),
            Some(6),
            None,
            Some(8)
        ]
    );
}

#[test]
fn offset() {
    let a = Bitmap::from([true, false, true, false, false, true, true, false, true]).sliced(1, 8);
    let a = Some(a.iter());
    let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let zip = ZipValidity::new(values.into_iter(), a);

    let a = zip.collect::<Vec<_>>();
    assert_eq!(
        a,
        vec![None, Some(1), None, None, Some(4), Some(5), None, Some(7)]
    );
}

#[test]
fn none() {
    let values = vec![0, 1, 2];
    let zip = ZipValidity::new(values.into_iter(), None::<BitmapIter>);

    let a = zip.collect::<Vec<_>>();
    assert_eq!(a, vec![Some(0), Some(1), Some(2)]);
}

#[test]
fn rev() {
    let a = Bitmap::from([true, false, true, false, false, true, true, false, true]).sliced(1, 8);
    let a = Some(a.iter());
    let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let zip = ZipValidity::new(values.into_iter(), a);

    let result = zip.rev().collect::<Vec<_>>();
    let expected = vec![None, Some(1), None, None, Some(4), Some(5), None, Some(7)]
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    assert_eq!(result, expected);
}
