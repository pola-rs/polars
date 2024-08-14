use super::*;

#[test]
fn test_datetime_as_lit() {
    let Expr::Alias(e, name) = datetime(Default::default()) else {
        panic!()
    };
    assert_eq!(name.as_ref(), "datetime");
    assert!(matches!(e.as_ref(), Expr::Literal(_)))
}

#[test]
fn test_duration_as_lit() {
    let Expr::Alias(e, name) = duration(Default::default()) else {
        panic!()
    };
    assert_eq!(name.as_ref(), "duration");
    assert!(matches!(e.as_ref(), Expr::Literal(_)))
}
