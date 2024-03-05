use arrow::array::growable::{Growable, GrowableBoolean};
use arrow::array::BooleanArray;

#[test]
fn test_bool() {
    let array = BooleanArray::from(vec![Some(false), Some(true), None, Some(false)]);

    let mut a = GrowableBoolean::new(vec![&array], false, 0);

    unsafe {
        a.extend(0, 1, 2);
    }
    assert_eq!(a.len(), 2);

    let result: BooleanArray = a.into();

    let expected = BooleanArray::from(vec![Some(true), None]);
    assert_eq!(result, expected);
}
