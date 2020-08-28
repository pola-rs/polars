pub trait Selection<'a> {
    fn to_selection_vec(self) -> Vec<&'a str>;
}

impl<'a> Selection<'a> for &'a str {
    fn to_selection_vec(self) -> Vec<&'a str> {
        vec![self]
    }
}

impl<'a> Selection<'a> for Vec<&'a str> {
    fn to_selection_vec(self) -> Vec<&'a str> {
        self
    }
}

impl<'a, T> Selection<'a> for &T
where
    T: AsRef<[&'a str]>,
{
    fn to_selection_vec(self) -> Vec<&'a str> {
        self.as_ref().iter().copied().collect()
    }
}

impl<'a> Selection<'a> for (&'a str, &'a str) {
    fn to_selection_vec(self) -> Vec<&'a str> {
        vec![self.0, self.1]
    }
}
impl<'a> Selection<'a> for (&'a str, &'a str, &'a str) {
    fn to_selection_vec(self) -> Vec<&'a str> {
        vec![self.0, self.1, self.2]
    }
}

impl<'a> Selection<'a> for (&'a str, &'a str, &'a str, &'a str) {
    fn to_selection_vec(self) -> Vec<&'a str> {
        vec![self.0, self.1, self.2, self.3]
    }
}

impl<'a> Selection<'a> for (&'a str, &'a str, &'a str, &'a str, &'a str) {
    fn to_selection_vec(self) -> Vec<&'a str> {
        vec![self.0, self.1, self.2, self.3, self.4]
    }
}

impl<'a> Selection<'a> for (&'a str, &'a str, &'a str, &'a str, &'a str, &'a str) {
    fn to_selection_vec(self) -> Vec<&'a str> {
        vec![self.0, self.1, self.2, self.3, self.4, self.5]
    }
}
