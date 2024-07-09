use arrow::array::PrimitiveArray;
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;
use either::Either;

#[test]
fn array_to_mutable() {
    let data = vec![1, 2, 3];
    let arr = PrimitiveArray::new(ArrowDataType::Int32, data.into(), None);

    // to mutable push and freeze again
    let mut mut_arr = arr.into_mut().unwrap_right();
    mut_arr.push(Some(5));
    let immut: PrimitiveArray<i32> = mut_arr.into();
    assert_eq!(immut.values().as_slice(), [1, 2, 3, 5]);

    // let's cause a realloc and see if miri is ok
    let mut mut_arr = immut.into_mut().unwrap_right();
    mut_arr.extend_constant(256, Some(9));
    let immut: PrimitiveArray<i32> = mut_arr.into();
    assert_eq!(immut.values().len(), 256 + 4);
}

#[test]
fn array_to_mutable_not_owned() {
    let data = vec![1, 2, 3];
    let arr = PrimitiveArray::new(ArrowDataType::Int32, data.into(), None);
    let arr2 = arr.clone();

    // to the `to_mutable` should fail and we should get back the original array
    match arr2.into_mut() {
        Either::Left(arr2) => {
            assert_eq!(arr, arr2);
        },
        _ => panic!(),
    }
}

#[test]
#[allow(clippy::redundant_clone)]
fn array_to_mutable_validity() {
    let data = vec![1, 2, 3];

    // both have a single reference should be ok
    let bitmap = Bitmap::from_iter([true, false, true]);
    let arr = PrimitiveArray::new(ArrowDataType::Int32, data.clone().into(), Some(bitmap));
    assert!(matches!(arr.into_mut(), Either::Right(_)));

    // now we clone the bitmap increasing the ref count
    let bitmap = Bitmap::from_iter([true, false, true]);
    let arr = PrimitiveArray::new(ArrowDataType::Int32, data.into(), Some(bitmap.clone()));
    assert!(matches!(arr.into_mut(), Either::Left(_)));
}
