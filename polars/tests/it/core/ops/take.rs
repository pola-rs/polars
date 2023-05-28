use super::*;

#[test]
fn test_list_take_nulls_and_empty() {
    unsafe {
        let a: &[i32] = &[];
        let a = Series::new("", a);
        let b = Series::new("", &[None, Some(a.clone())]);
        let mut iter = [Some(0), Some(1usize), None].iter().copied();
        let out = b.take_opt_iter_unchecked(&mut iter);
        let expected = Series::new("", &[None, Some(a.clone()), None]);
        assert!(out.series_equal_missing(&expected))
    }
}
