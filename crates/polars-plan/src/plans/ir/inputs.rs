use std::iter;

use super::*;

impl IR {
    /// Returns a node with updated expressions.
    ///
    /// Panics if the expression count doesn't match
    /// [`Self::exprs`]/[`Self::exprs_mut`]/[`Self::copy_exprs`].
    pub fn with_exprs<E>(mut self, exprs: E) -> Self
    where
        E: IntoIterator<Item = ExprIR>,
    {
        let mut exprs_mut = self.exprs_mut();
        let mut new_exprs = exprs.into_iter();

        for (expr, new_expr) in exprs_mut.by_ref().zip(new_exprs.by_ref()) {
            *expr = new_expr;
        }

        assert!(exprs_mut.next().is_none(), "not enough exprs");
        assert!(new_exprs.next().is_none(), "too many exprs");

        drop(exprs_mut);

        self
    }

    /// Returns a node with updated inputs.
    ///
    /// Panics if the input count doesn't match
    /// [`Self::inputs`]/[`Self::inputs_mut`]/[`Self::copy_inputs`]/[`Self::get_inputs`].
    pub fn with_inputs<I>(mut self, inputs: I) -> Self
    where
        I: IntoIterator<Item = Node>,
    {
        let mut inputs_mut = self.inputs_mut();
        let mut new_inputs = inputs.into_iter();

        for (input, new_input) in inputs_mut.by_ref().zip(new_inputs.by_ref()) {
            *input = new_input;
        }

        assert!(inputs_mut.next().is_none(), "not enough inputs");
        assert!(new_inputs.next().is_none(), "too many inputs");

        drop(inputs_mut);

        self
    }

    pub fn exprs(&'_ self) -> Exprs<'_> {
        use IR::*;
        match self {
            Slice { .. } => Exprs::Empty,
            Cache { .. } => Exprs::Empty,
            Distinct { .. } => Exprs::Empty,
            Union { .. } => Exprs::Empty,
            MapFunction { .. } => Exprs::Empty,
            DataFrameScan { .. } => Exprs::Empty,
            HConcat { .. } => Exprs::Empty,
            ExtContext { .. } => Exprs::Empty,
            SimpleProjection { .. } => Exprs::Empty,
            SinkMultiple { .. } => Exprs::Empty,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => Exprs::Empty,

            #[cfg(feature = "python")]
            PythonScan { options } => match &options.predicate {
                PythonPredicate::Polars(predicate) => Exprs::single(predicate),
                _ => Exprs::Empty,
            },

            Scan { predicate, .. } => match predicate {
                Some(predicate) => Exprs::single(predicate),
                _ => Exprs::Empty,
            },

            Filter { predicate, .. } => Exprs::single(predicate),

            Sort { by_column, .. } => Exprs::slice(by_column),
            Select { expr, .. } => Exprs::slice(expr),
            HStack { exprs, .. } => Exprs::slice(exprs),

            GroupBy { keys, aggs, .. } => Exprs::double_slice(keys, aggs),

            Join {
                left_on,
                right_on,
                options,
                ..
            } => match &options.options {
                Some(JoinTypeOptionsIR::CrossAndFilter { predicate }) => Exprs::Boxed(Box::new(
                    left_on
                        .iter()
                        .chain(right_on.iter())
                        .chain(iter::once(predicate)),
                )),
                _ => Exprs::double_slice(left_on, right_on),
            },

            Sink { payload, .. } => match payload {
                SinkTypeIR::Memory => Exprs::Empty,
                SinkTypeIR::File(_) => Exprs::Empty,
                SinkTypeIR::Partition(p) => {
                    let key_iter = match &p.variant {
                        PartitionVariantIR::Parted { key_exprs, .. }
                        | PartitionVariantIR::ByKey { key_exprs, .. } => key_exprs.iter(),
                        _ => [].iter(),
                    };
                    let sort_by_iter = match &p.per_partition_sort_by {
                        Some(sort_by) => sort_by.iter(),
                        _ => [].iter(),
                    }
                    .map(|s| &s.expr);
                    Exprs::Boxed(Box::new(key_iter.chain(sort_by_iter)))
                },
            },

            Invalid => unreachable!(),
        }
    }

    pub fn exprs_mut(&'_ mut self) -> ExprsMut<'_> {
        use IR::*;
        match self {
            Slice { .. } => ExprsMut::Empty,
            Cache { .. } => ExprsMut::Empty,
            Distinct { .. } => ExprsMut::Empty,
            Union { .. } => ExprsMut::Empty,
            MapFunction { .. } => ExprsMut::Empty,
            DataFrameScan { .. } => ExprsMut::Empty,
            HConcat { .. } => ExprsMut::Empty,
            ExtContext { .. } => ExprsMut::Empty,
            SimpleProjection { .. } => ExprsMut::Empty,
            SinkMultiple { .. } => ExprsMut::Empty,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => ExprsMut::Empty,

            #[cfg(feature = "python")]
            PythonScan { options } => match &mut options.predicate {
                PythonPredicate::Polars(predicate) => ExprsMut::single(predicate),
                _ => ExprsMut::Empty,
            },

            Scan { predicate, .. } => match predicate {
                Some(predicate) => ExprsMut::single(predicate),
                _ => ExprsMut::Empty,
            },

            Filter { predicate, .. } => ExprsMut::single(predicate),

            Sort { by_column, .. } => ExprsMut::slice(by_column),
            Select { expr, .. } => ExprsMut::slice(expr),
            HStack { exprs, .. } => ExprsMut::slice(exprs),

            GroupBy { keys, aggs, .. } => ExprsMut::double_slice(keys, aggs),

            Join {
                left_on,
                right_on,
                options,
                ..
            } => match Arc::make_mut(options).options.as_mut() {
                Some(JoinTypeOptionsIR::CrossAndFilter { predicate }) => ExprsMut::Boxed(Box::new(
                    left_on
                        .iter_mut()
                        .chain(right_on.iter_mut())
                        .chain(iter::once(predicate)),
                )),
                _ => ExprsMut::double_slice(left_on, right_on),
            },

            Sink { payload, .. } => match payload {
                SinkTypeIR::Memory => ExprsMut::Empty,
                SinkTypeIR::File(_) => ExprsMut::Empty,
                SinkTypeIR::Partition(p) => {
                    let key_iter = match &mut p.variant {
                        PartitionVariantIR::Parted { key_exprs, .. }
                        | PartitionVariantIR::ByKey { key_exprs, .. } => key_exprs.iter_mut(),
                        _ => [].iter_mut(),
                    };
                    let sort_by_iter = match &mut p.per_partition_sort_by {
                        Some(sort_by) => sort_by.iter_mut(),
                        _ => [].iter_mut(),
                    }
                    .map(|s| &mut s.expr);
                    ExprsMut::Boxed(Box::new(key_iter.chain(sort_by_iter)))
                },
            },

            Invalid => unreachable!(),
        }
    }

    /// Copy the exprs in this LP node to an existing container.
    pub fn copy_exprs<T>(&self, container: &mut T)
    where
        T: Extend<ExprIR>,
    {
        container.extend(self.exprs().cloned())
    }

    pub fn inputs(&'_ self) -> Inputs<'_> {
        use IR::*;
        match self {
            Union { inputs, .. } | HConcat { inputs, .. } | SinkMultiple { inputs } => {
                Inputs::slice(inputs)
            },
            Slice { input, .. } => Inputs::single(*input),
            Filter { input, .. } => Inputs::single(*input),
            Select { input, .. } => Inputs::single(*input),
            SimpleProjection { input, .. } => Inputs::single(*input),
            Sort { input, .. } => Inputs::single(*input),
            Cache { input, .. } => Inputs::single(*input),
            GroupBy { input, .. } => Inputs::single(*input),
            Join {
                input_left,
                input_right,
                ..
            } => Inputs::double(*input_left, *input_right),
            HStack { input, .. } => Inputs::single(*input),
            Distinct { input, .. } => Inputs::single(*input),
            MapFunction { input, .. } => Inputs::single(*input),
            Sink { input, .. } => Inputs::single(*input),
            ExtContext {
                input, contexts, ..
            } => Inputs::Boxed(Box::new(iter::once(*input).chain(contexts.iter().copied()))),
            Scan { .. } => Inputs::Empty,
            DataFrameScan { .. } => Inputs::Empty,
            #[cfg(feature = "python")]
            PythonScan { .. } => Inputs::Empty,
            #[cfg(feature = "merge_sorted")]
            MergeSorted {
                input_left,
                input_right,
                ..
            } => Inputs::double(*input_left, *input_right),
            Invalid => unreachable!(),
        }
    }

    pub fn inputs_mut(&'_ mut self) -> InputsMut<'_> {
        use IR::*;
        match self {
            Union { inputs, .. } | HConcat { inputs, .. } | SinkMultiple { inputs } => {
                InputsMut::slice(inputs)
            },
            Slice { input, .. } => InputsMut::single(input),
            Filter { input, .. } => InputsMut::single(input),
            Select { input, .. } => InputsMut::single(input),
            SimpleProjection { input, .. } => InputsMut::single(input),
            Sort { input, .. } => InputsMut::single(input),
            Cache { input, .. } => InputsMut::single(input),
            GroupBy { input, .. } => InputsMut::single(input),
            Join {
                input_left,
                input_right,
                ..
            } => InputsMut::double(input_left, input_right),
            HStack { input, .. } => InputsMut::single(input),
            Distinct { input, .. } => InputsMut::single(input),
            MapFunction { input, .. } => InputsMut::single(input),
            Sink { input, .. } => InputsMut::single(input),
            ExtContext {
                input, contexts, ..
            } => InputsMut::Boxed(Box::new(iter::once(input).chain(contexts.iter_mut()))),
            Scan { .. } => InputsMut::Empty,
            DataFrameScan { .. } => InputsMut::Empty,
            #[cfg(feature = "python")]
            PythonScan { .. } => InputsMut::Empty,
            #[cfg(feature = "merge_sorted")]
            MergeSorted {
                input_left,
                input_right,
                ..
            } => InputsMut::double(input_left, input_right),
            Invalid => unreachable!(),
        }
    }

    /// Push inputs of the LP in of this node to an existing container.
    /// Most plans have typically one input. A join has two and a scan (CsvScan)
    /// or an in-memory DataFrame has none. A Union has multiple.
    pub fn copy_inputs<T>(&self, container: &mut T)
    where
        T: Extend<Node>,
    {
        container.extend(self.inputs())
    }

    pub fn get_inputs(&self) -> UnitVec<Node> {
        self.inputs().collect()
    }

    pub(crate) fn get_input(&self) -> Option<Node> {
        self.inputs().next()
    }
}

pub enum Inputs<'a> {
    Empty,
    Single(iter::Once<Node>),
    Double(std::array::IntoIter<Node, 2>),
    Slice(iter::Copied<std::slice::Iter<'a, Node>>),
    Boxed(Box<dyn Iterator<Item = Node> + 'a>),
}

impl<'a> Inputs<'a> {
    fn single(node: Node) -> Self {
        Self::Single(iter::once(node))
    }

    fn double(left: Node, right: Node) -> Self {
        Self::Double([left, right].into_iter())
    }

    fn slice(inputs: &'a [Node]) -> Self {
        Self::Slice(inputs.iter().copied())
    }
}

impl<'a> Iterator for Inputs<'a> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Single(it) => it.next(),
            Self::Double(it) => it.next(),
            Self::Slice(it) => it.next(),
            Self::Boxed(it) => it.next(),
        }
    }
}

pub enum InputsMut<'a> {
    Empty,
    Single(iter::Once<&'a mut Node>),
    Double(std::array::IntoIter<&'a mut Node, 2>),
    Slice(std::slice::IterMut<'a, Node>),
    Boxed(Box<dyn Iterator<Item = &'a mut Node> + 'a>),
}

impl<'a> InputsMut<'a> {
    fn single(node: &'a mut Node) -> Self {
        Self::Single(iter::once(node))
    }

    fn double(left: &'a mut Node, right: &'a mut Node) -> Self {
        Self::Double([left, right].into_iter())
    }

    fn slice(inputs: &'a mut [Node]) -> Self {
        Self::Slice(inputs.iter_mut())
    }
}

impl<'a> Iterator for InputsMut<'a> {
    type Item = &'a mut Node;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Single(it) => it.next(),
            Self::Double(it) => it.next(),
            Self::Slice(it) => it.next(),
            Self::Boxed(it) => it.next(),
        }
    }
}

pub enum Exprs<'a> {
    Empty,
    Single(iter::Once<&'a ExprIR>),
    Slice(std::slice::Iter<'a, ExprIR>),
    DoubleSlice(iter::Chain<std::slice::Iter<'a, ExprIR>, std::slice::Iter<'a, ExprIR>>),
    Boxed(Box<dyn Iterator<Item = &'a ExprIR> + 'a>),
}

impl<'a> Exprs<'a> {
    fn single(expr: &'a ExprIR) -> Self {
        Self::Single(iter::once(expr))
    }

    fn slice(inputs: &'a [ExprIR]) -> Self {
        Self::Slice(inputs.iter())
    }

    fn double_slice(left: &'a [ExprIR], right: &'a [ExprIR]) -> Self {
        Self::DoubleSlice(left.iter().chain(right.iter()))
    }
}

impl<'a> Iterator for Exprs<'a> {
    type Item = &'a ExprIR;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Single(it) => it.next(),
            Self::Slice(it) => it.next(),
            Self::DoubleSlice(it) => it.next(),
            Self::Boxed(it) => it.next(),
        }
    }
}

pub enum ExprsMut<'a> {
    Empty,
    Single(iter::Once<&'a mut ExprIR>),
    Slice(std::slice::IterMut<'a, ExprIR>),
    DoubleSlice(iter::Chain<std::slice::IterMut<'a, ExprIR>, std::slice::IterMut<'a, ExprIR>>),
    Boxed(Box<dyn Iterator<Item = &'a mut ExprIR> + 'a>),
}

impl<'a> ExprsMut<'a> {
    fn single(expr: &'a mut ExprIR) -> Self {
        Self::Single(iter::once(expr))
    }

    fn slice(inputs: &'a mut [ExprIR]) -> Self {
        Self::Slice(inputs.iter_mut())
    }

    fn double_slice(left: &'a mut [ExprIR], right: &'a mut [ExprIR]) -> Self {
        Self::DoubleSlice(left.iter_mut().chain(right.iter_mut()))
    }
}

impl<'a> Iterator for ExprsMut<'a> {
    type Item = &'a mut ExprIR;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Single(it) => it.next(),
            Self::Slice(it) => it.next(),
            Self::DoubleSlice(it) => it.next(),
            Self::Boxed(it) => it.next(),
        }
    }
}
