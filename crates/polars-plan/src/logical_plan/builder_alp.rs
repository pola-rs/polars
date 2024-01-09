use std::borrow::Cow;

use super::builder_functions::*;
use super::*;
use crate::logical_plan::projection_expr::ProjectionExprs;

pub struct ALogicalPlanBuilder<'a> {
    root: Node,
    expr_arena: &'a mut Arena<AExpr>,
    lp_arena: &'a mut Arena<ALogicalPlan>,
}

impl<'a> ALogicalPlanBuilder<'a> {
    pub(crate) fn new(
        root: Node,
        expr_arena: &'a mut Arena<AExpr>,
        lp_arena: &'a mut Arena<ALogicalPlan>,
    ) -> Self {
        ALogicalPlanBuilder {
            root,
            expr_arena,
            lp_arena,
        }
    }

    pub(crate) fn from_lp(
        lp: ALogicalPlan,
        expr_arena: &'a mut Arena<AExpr>,
        lp_arena: &'a mut Arena<ALogicalPlan>,
    ) -> Self {
        let root = lp_arena.add(lp);
        ALogicalPlanBuilder {
            root,
            expr_arena,
            lp_arena,
        }
    }

    fn add_alp(self, lp: ALogicalPlan) -> Self {
        let node = self.lp_arena.add(lp);
        ALogicalPlanBuilder::new(node, self.expr_arena, self.lp_arena)
    }

    pub fn project(self, exprs: Vec<Node>, options: ProjectionOptions) -> Self {
        let input_schema = self.lp_arena.get(self.root).schema(self.lp_arena);
        let schema = aexprs_to_schema(&exprs, &input_schema, Context::Default, self.expr_arena);

        // if len == 0, no projection has to be done. This is a select all operation.
        if !exprs.is_empty() {
            let lp = ALogicalPlan::Projection {
                expr: exprs.into(),
                input: self.root,
                schema: Arc::new(schema),
                options,
            };
            let node = self.lp_arena.add(lp);
            ALogicalPlanBuilder::new(node, self.expr_arena, self.lp_arena)
        } else {
            self
        }
    }

    pub fn build(self) -> ALogicalPlan {
        if self.root.0 == self.lp_arena.len() {
            self.lp_arena.pop().unwrap()
        } else {
            self.lp_arena.take(self.root)
        }
    }

    pub(crate) fn schema(&'a self) -> Cow<'a, SchemaRef> {
        self.lp_arena.get(self.root).schema(self.lp_arena)
    }

    pub(crate) fn with_columns(self, exprs: Vec<Node>, options: ProjectionOptions) -> Self {
        let schema = self.schema();
        let mut new_schema = (**schema).clone();

        for e in &exprs {
            let field = self
                .expr_arena
                .get(*e)
                .to_field(&schema, Context::Default, self.expr_arena)
                .unwrap();

            new_schema.with_column(field.name().clone(), field.data_type().clone());
        }

        let lp = ALogicalPlan::HStack {
            input: self.root,
            exprs: ProjectionExprs::new(exprs),
            schema: Arc::new(new_schema),
            options,
        };
        self.add_alp(lp)
    }

    // call this if the schema needs to be updated
    pub(crate) fn explode(self, columns: Arc<[Arc<str>]>) -> Self {
        let mut schema = (*self.schema().into_owned()).clone();
        explode_schema(&mut schema, &columns).unwrap();

        let lp = ALogicalPlan::MapFunction {
            input: self.root,
            function: FunctionNode::Explode {
                columns,
                schema: Arc::new(schema),
            },
        };
        self.add_alp(lp)
    }

    pub fn group_by(
        self,
        keys: Vec<Node>,
        aggs: Vec<Node>,
        apply: Option<Arc<dyn DataFrameUdf>>,
        maintain_order: bool,
        options: Arc<GroupbyOptions>,
    ) -> Self {
        let current_schema = self.schema();
        // TODO! add this line if LogicalPlan is dropped in favor of ALogicalPlan
        // let aggs = rewrite_projections(aggs, current_schema);

        let mut schema =
            aexprs_to_schema(&keys, &current_schema, Context::Default, self.expr_arena);

        #[cfg(feature = "dynamic_group_by")]
        {
            if let Some(options) = options.rolling.as_ref() {
                let name = &options.index_column;
                let dtype = current_schema.get(name).unwrap();
                schema.with_column(name.clone(), dtype.clone());
            } else if let Some(options) = options.dynamic.as_ref() {
                let name = &options.index_column;
                let dtype = current_schema.get(name).unwrap();
                if options.include_boundaries {
                    schema.with_column("_lower_boundary".into(), dtype.clone());
                    schema.with_column("_upper_boundary".into(), dtype.clone());
                }
                schema.with_column(name.clone(), dtype.clone());
            }
        }

        let agg_schema = aexprs_to_schema(
            &aggs,
            &current_schema,
            Context::Aggregation,
            self.expr_arena,
        );
        schema.merge(agg_schema);

        let lp = ALogicalPlan::Aggregate {
            input: self.root,
            keys,
            aggs,
            schema: Arc::new(schema),
            apply,
            maintain_order,
            options,
        };
        self.add_alp(lp)
    }

    pub fn join(
        self,
        other: Node,
        left_on: Vec<Node>,
        right_on: Vec<Node>,
        options: Arc<JoinOptions>,
    ) -> Self {
        let schema_left = self.schema();
        let schema_right = self.lp_arena.get(other).schema(self.lp_arena);

        let left_on_exprs = left_on
            .iter()
            .map(|node| node_to_expr(*node, self.expr_arena))
            .collect::<Vec<_>>();
        let right_on_exprs = right_on
            .iter()
            .map(|node| node_to_expr(*node, self.expr_arena))
            .collect::<Vec<_>>();

        let schema = det_join_schema(
            &schema_left,
            &schema_right,
            &left_on_exprs,
            &right_on_exprs,
            &options,
        )
        .unwrap();

        let lp = ALogicalPlan::Join {
            input_left: self.root,
            input_right: other,
            schema,
            left_on,
            right_on,
            options,
        };

        self.add_alp(lp)
    }

    pub fn melt(self, args: Arc<MeltArgs>) -> Self {
        let schema = self.schema();
        let schema = det_melt_schema(&args, &schema);
        let lp = ALogicalPlan::MapFunction {
            input: self.root,
            function: FunctionNode::Melt { args, schema },
        };
        self.add_alp(lp)
    }

    pub fn row_count(self, name: Arc<str>, offset: Option<IdxSize>) -> Self {
        let mut schema = self.schema().into_owned();
        let schema_mut = Arc::make_mut(&mut schema);
        row_index_schema(schema_mut, name.as_ref());

        let lp = ALogicalPlan::MapFunction {
            input: self.root,
            function: FunctionNode::RowIndex {
                name,
                offset,
                schema,
            },
        };
        self.add_alp(lp)
    }
}
