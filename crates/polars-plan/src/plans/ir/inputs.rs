use std::iter;

use polars_buffer::Buffer;
use polars_utils::itertools::Itertools;

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
        for (expr, new_expr) in self.exprs_mut().zip_eq(exprs) {
            *expr = new_expr;
        }
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
        for (input, new_input) in self.inputs_mut().zip_eq(inputs) {
            *input = new_input;
        }
        self
    }

    pub fn exprs(&'_ self) -> Exprs<'_> {
        use IR::*;
        match self {
            Slice { .. }
            | Cache { .. }
            | Distinct { .. }
            | Union { .. }
            | MapFunction { .. }
            | DataFrameScan { .. }
            | HConcat { .. }
            | SimpleProjection { .. }
            | SinkMultiple { .. }
            | Gather { .. } => Exprs::Empty,

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

            Join { options, .. } => options.options.exprs(),

            Sink { payload, .. } => match payload {
                SinkTypeIR::Memory => Exprs::Empty,
                SinkTypeIR::Callback(_) => Exprs::Empty,

                SinkTypeIR::File(_) => Exprs::Empty,

                SinkTypeIR::Partitioned(PartitionedSinkOptionsIR {
                    partition_strategy, ..
                }) => match partition_strategy {
                    PartitionStrategyIR::Keyed {
                        keys,
                        include_keys: _,
                        keys_pre_grouped: _,
                    } => Exprs::Slice(keys.iter()),
                    PartitionStrategyIR::FileSize => Exprs::Empty,
                },
            },

            UnoptimizedDispatch { .. } => Exprs::Empty,
            Resolver { filters, .. } => Exprs::slice(filters),
            Invalid => unreachable!(),
        }
    }

    pub fn exprs_mut(&'_ mut self) -> ExprsMut<'_> {
        use IR::*;
        match self {
            Slice { .. }
            | Cache { .. }
            | Distinct { .. }
            | Union { .. }
            | MapFunction { .. }
            | DataFrameScan { .. }
            | HConcat { .. }
            | SimpleProjection { .. }
            | SinkMultiple { .. }
            | Gather { .. } => ExprsMut::Empty,
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

            Join { options, .. } => Arc::make_mut(options).options.exprs_mut(),

            Sink { payload, .. } => match payload {
                SinkTypeIR::Memory => ExprsMut::Empty,
                SinkTypeIR::Callback(_) => ExprsMut::Empty,

                SinkTypeIR::File(_) => ExprsMut::Empty,

                SinkTypeIR::Partitioned(PartitionedSinkOptionsIR {
                    partition_strategy, ..
                }) => match partition_strategy {
                    PartitionStrategyIR::Keyed {
                        keys,
                        include_keys: _,
                        keys_pre_grouped: _,
                    } => ExprsMut::Slice(keys.iter_mut()),
                    PartitionStrategyIR::FileSize => ExprsMut::Empty,
                },
            },

            UnoptimizedDispatch { .. } => ExprsMut::Empty,
            Resolver { filters, .. } => {
                if filters.get_mut_slice().is_none() {
                    *filters = Buffer::from_iter(filters.iter().cloned());
                }

                ExprsMut::slice(filters.get_mut_slice().unwrap())
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

    pub fn inputs(&self) -> Inputs<'_> {
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
            Gather { input, idxs, .. } => Inputs::double(*input, *idxs),
            HStack { input, .. } => Inputs::single(*input),
            Distinct { input, .. } => Inputs::single(*input),
            MapFunction { input, .. } => Inputs::single(*input),
            Sink { input, .. } => Inputs::single(*input),
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
            UnoptimizedDispatch { inputs, .. } => Inputs::slice(inputs),
            Resolver { resolved_ir, .. } => Inputs::Slice(resolved_ir.as_slice().iter().copied()),
            Invalid => unreachable!(),
        }
    }

    pub fn inputs_mut(&mut self) -> InputsMut<'_> {
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
            Gather { input, idxs, .. } => InputsMut::double(input, idxs),
            HStack { input, .. } => InputsMut::single(input),
            Distinct { input, .. } => InputsMut::single(input),
            MapFunction { input, .. } => InputsMut::single(input),
            Sink { input, .. } => InputsMut::single(input),
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
            UnoptimizedDispatch { inputs, .. } => InputsMut::slice(inputs),
            Resolver { resolved_ir, .. } => InputsMut::slice(resolved_ir.as_mut_slice()),
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
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Single(it) => it.nth(n),
            Self::Double(it) => it.nth(n),
            Self::Slice(it) => it.nth(n),
        }
    }
}

pub enum InputsMut<'a> {
    Empty,
    Single(iter::Once<&'a mut Node>),
    Double(std::array::IntoIter<&'a mut Node, 2>),
    Slice(std::slice::IterMut<'a, Node>),
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
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Single(it) => it.nth(n),
            Self::Double(it) => it.nth(n),
            Self::Slice(it) => it.nth(n),
        }
    }
}

/// One side of a paired key list, e.g. every `l` of `[(l, r), ..]`.
pub type PairSide<'a> =
    iter::Map<std::slice::Iter<'a, (ExprIR, ExprIR)>, fn(&'a (ExprIR, ExprIR)) -> &'a ExprIR>;

pub enum Exprs<'a> {
    Empty,
    Single(iter::Once<&'a ExprIR>),
    Slice(std::slice::Iter<'a, ExprIR>),
    DoubleSlice(iter::Chain<std::slice::Iter<'a, ExprIR>, std::slice::Iter<'a, ExprIR>>),
    PairSide(PairSide<'a>),
    /// Every left-hand side followed by every right-hand side.
    PairSides(iter::Chain<PairSide<'a>, PairSide<'a>>),
    Boxed(Box<dyn Iterator<Item = &'a ExprIR> + 'a>),
}

impl<'a> Exprs<'a> {
    pub(crate) fn single(expr: &'a ExprIR) -> Self {
        Self::Single(iter::once(expr))
    }

    pub(crate) fn slice(inputs: &'a [ExprIR]) -> Self {
        Self::Slice(inputs.iter())
    }

    pub(crate) fn double_slice(left: &'a [ExprIR], right: &'a [ExprIR]) -> Self {
        Self::DoubleSlice(left.iter().chain(right.iter()))
    }

    pub(crate) fn pair_lhs(on: &'a [(ExprIR, ExprIR)]) -> Self {
        Self::PairSide(on.iter().map(pair_lhs))
    }

    pub(crate) fn pair_rhs(on: &'a [(ExprIR, ExprIR)]) -> Self {
        Self::PairSide(on.iter().map(pair_rhs))
    }

    /// All left-hand sides, then all right-hand sides.
    pub(crate) fn pair_sides(on: &'a [(ExprIR, ExprIR)]) -> Self {
        Self::PairSides(
            on.iter()
                .map(pair_lhs as _)
                .chain(on.iter().map(pair_rhs as _)),
        )
    }
}

fn pair_lhs((lhs, _): &(ExprIR, ExprIR)) -> &ExprIR {
    lhs
}

fn pair_rhs((_, rhs): &(ExprIR, ExprIR)) -> &ExprIR {
    rhs
}

impl<'a> Iterator for Exprs<'a> {
    type Item = &'a ExprIR;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Single(it) => it.next(),
            Self::Slice(it) => it.next(),
            Self::DoubleSlice(it) => it.next(),
            Self::PairSide(it) => it.next(),
            Self::PairSides(it) => it.next(),
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
    pub(crate) fn single(expr: &'a mut ExprIR) -> Self {
        Self::Single(iter::once(expr))
    }

    pub(crate) fn slice(inputs: &'a mut [ExprIR]) -> Self {
        Self::Slice(inputs.iter_mut())
    }

    pub(crate) fn double_slice(left: &'a mut [ExprIR], right: &'a mut [ExprIR]) -> Self {
        Self::DoubleSlice(left.iter_mut().chain(right.iter_mut()))
    }

    /// All left-hand sides, then all right-hand sides. Must match [`Exprs::pair_sides`].
    ///
    /// Collects the borrows because two disjoint `&mut` iterators over one slice of pairs
    /// cannot be built in safe Rust.
    pub(crate) fn pair_sides(on: &'a mut [(ExprIR, ExprIR)]) -> Self {
        let (lhs, rhs): (Vec<_>, Vec<_>) = on.iter_mut().map(|(l, r)| (l, r)).unzip();
        Self::Boxed(Box::new(lhs.into_iter().chain(rhs)))
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
