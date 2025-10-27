pub mod mean;

#[cfg(test)]
fn assert_all_close<X, Y, T>(xs: X, ys: Y, tolerance: T)
where
    X: IntoIterator<Item = Option<T>>,
    Y: IntoIterator<Item = Option<T>>,
    T: std::ops::Sub<T, Output = T> + num_traits::Signed + std::cmp::PartialOrd,
{
    let xs = xs.into_iter();
    let ys = ys.into_iter();

    assert_eq!(xs.size_hint().1.unwrap(), ys.size_hint().1.unwrap());

    assert!(xs.zip(ys).all(|(x, y)| match (x, y) {
        (Some(x), Some(y)) => (x - y).abs() < tolerance,
        (None, None) => true,
        _ => false,
    }))
}
