#[allow(clippy::wrong_self_convention)]
/// Allow selection by:
///
/// &str => df.select("my-column"),
/// (&str)" => df.select(("col_1", "col_2")),
/// Vec<&str)" => df.select(vec!["col_a", "col_b"]),
pub trait Selection<'a, S> {
    fn to_selection_vec(self) -> Vec<&'a str>;

    // a single column is selected
    fn single(&self) -> Option<&'a str> {
        None
    }
}

impl<'a> Selection<'a, &str> for &'a str {
    fn to_selection_vec(self) -> Vec<&'a str> {
        vec![self]
    }
    fn single(&self) -> Option<&'a str> {
        Some(self)
    }
}

impl<'a> Selection<'a, &str> for Vec<&'a str> {
    fn to_selection_vec(self) -> Vec<&'a str> {
        self
    }
}

impl<'a, T, S: 'a> Selection<'a, S> for &'a T
where
    T: AsRef<[S]>,
    S: AsRef<str>,
{
    fn to_selection_vec(self) -> Vec<&'a str> {
        self.as_ref().iter().map(|s| s.as_ref()).collect()
    }
}

impl<'a> Selection<'a, &str> for (&'a str, &'a str) {
    fn to_selection_vec(self) -> Vec<&'a str> {
        vec![self.0, self.1]
    }
}
impl<'a> Selection<'a, &str> for (&'a str, &'a str, &'a str) {
    fn to_selection_vec(self) -> Vec<&'a str> {
        vec![self.0, self.1, self.2]
    }
}

impl<'a> Selection<'a, &str> for (&'a str, &'a str, &'a str, &'a str) {
    fn to_selection_vec(self) -> Vec<&'a str> {
        vec![self.0, self.1, self.2, self.3]
    }
}

impl<'a> Selection<'a, &str> for (&'a str, &'a str, &'a str, &'a str, &'a str) {
    fn to_selection_vec(self) -> Vec<&'a str> {
        vec![self.0, self.1, self.2, self.3, self.4]
    }
}

impl<'a> Selection<'a, &str> for (&'a str, &'a str, &'a str, &'a str, &'a str, &'a str) {
    fn to_selection_vec(self) -> Vec<&'a str> {
        vec![self.0, self.1, self.2, self.3, self.4, self.5]
    }
}
