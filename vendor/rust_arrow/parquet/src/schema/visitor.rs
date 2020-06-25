// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::basic::{LogicalType, Repetition};
use crate::errors::ParquetError::General;
use crate::errors::Result;
use crate::schema::types::{Type, TypePtr};

/// A utility trait to help user to traverse against parquet type.
pub trait TypeVisitor<R, C> {
    /// Called when a primitive type hit.
    fn visit_primitive(&mut self, primitive_type: TypePtr, context: C) -> Result<R>;

    /// Default implementation when visiting a list.
    ///
    /// It checks list type definition and calls `visit_list_with_item` with extracted
    /// item type.
    ///
    /// To fully understand this algorithm, please refer to
    /// [parquet doc](https://github.com/apache/parquet-format/blob/master/LogicalTypes.md).
    fn visit_list(&mut self, list_type: TypePtr, context: C) -> Result<R> {
        match list_type.as_ref() {
            Type::PrimitiveType { .. } => panic!(
                "{:?} is a list type and can't be processed as primitive.",
                list_type
            ),
            Type::GroupType {
                basic_info: _,
                fields,
            } if fields.len() == 1 => {
                let list_item = fields.first().unwrap();

                match list_item.as_ref() {
                    Type::PrimitiveType { .. } => {
                        if list_item.get_basic_info().repetition() == Repetition::REPEATED
                        {
                            self.visit_list_with_item(
                                list_type.clone(),
                                list_item,
                                context,
                            )
                        } else {
                            Err(General(
                                "Primitive element type of list must be repeated."
                                    .to_string(),
                            ))
                        }
                    }
                    Type::GroupType {
                        basic_info: _,
                        fields,
                    } => {
                        if fields.len() == 1
                            && list_item.name() != "array"
                            && list_item.name() != format!("{}_tuple", list_type.name())
                        {
                            self.visit_list_with_item(
                                list_type.clone(),
                                fields.first().unwrap(),
                                context,
                            )
                        } else {
                            self.visit_list_with_item(
                                list_type.clone(),
                                list_item,
                                context,
                            )
                        }
                    }
                }
            }
            _ => Err(General(
                "Group element type of list can only contain one field.".to_string(),
            )),
        }
    }

    /// Called when a struct type hit.
    fn visit_struct(&mut self, struct_type: TypePtr, context: C) -> Result<R>;

    /// Called when a map type hit.
    fn visit_map(&mut self, map_type: TypePtr, context: C) -> Result<R>;

    /// A utility method which detects input type and calls corresponding method.
    fn dispatch(&mut self, cur_type: TypePtr, context: C) -> Result<R> {
        if cur_type.is_primitive() {
            self.visit_primitive(cur_type, context)
        } else {
            match cur_type.get_basic_info().logical_type() {
                LogicalType::LIST => self.visit_list(cur_type, context),
                LogicalType::MAP | LogicalType::MAP_KEY_VALUE => {
                    self.visit_map(cur_type, context)
                }
                _ => self.visit_struct(cur_type, context),
            }
        }
    }

    /// Called by `visit_list`.
    fn visit_list_with_item(
        &mut self,
        list_type: TypePtr,
        item_type: &Type,
        context: C,
    ) -> Result<R>;
}

#[cfg(test)]
mod tests {
    use super::TypeVisitor;
    use crate::basic::Type as PhysicalType;
    use crate::errors::Result;
    use crate::schema::parser::parse_message_type;
    use crate::schema::types::{Type, TypePtr};
    use std::rc::Rc;

    struct TestVisitorContext {}
    struct TestVisitor {
        primitive_visited: bool,
        struct_visited: bool,
        list_visited: bool,
        root_type: TypePtr,
    }

    impl TypeVisitor<bool, TestVisitorContext> for TestVisitor {
        fn visit_primitive(
            &mut self,
            primitive_type: TypePtr,
            _context: TestVisitorContext,
        ) -> Result<bool> {
            assert_eq!(
                self.get_field_by_name(primitive_type.name()).as_ref(),
                primitive_type.as_ref()
            );
            self.primitive_visited = true;
            Ok(true)
        }

        fn visit_struct(
            &mut self,
            struct_type: TypePtr,
            _context: TestVisitorContext,
        ) -> Result<bool> {
            assert_eq!(
                self.get_field_by_name(struct_type.name()).as_ref(),
                struct_type.as_ref()
            );
            self.struct_visited = true;
            Ok(true)
        }

        fn visit_map(
            &mut self,
            _map_type: TypePtr,
            _context: TestVisitorContext,
        ) -> Result<bool> {
            unimplemented!()
        }

        fn visit_list_with_item(
            &mut self,
            list_type: TypePtr,
            item_type: &Type,
            _context: TestVisitorContext,
        ) -> Result<bool> {
            assert_eq!(
                self.get_field_by_name(list_type.name()).as_ref(),
                list_type.as_ref()
            );
            assert_eq!("element", item_type.name());
            assert_eq!(PhysicalType::INT32, item_type.get_physical_type());
            self.list_visited = true;
            Ok(true)
        }
    }

    impl TestVisitor {
        fn new(root: TypePtr) -> Self {
            Self {
                primitive_visited: false,
                struct_visited: false,
                list_visited: false,
                root_type: root,
            }
        }

        fn get_field_by_name(&self, name: &str) -> TypePtr {
            self.root_type
                .get_fields()
                .iter()
                .find(|t| t.name() == name)
                .map(|t| t.clone())
                .unwrap()
        }
    }

    #[test]
    fn test_visitor() {
        let message_type = "
          message spark_schema {
            REQUIRED INT32 a;
            OPTIONAL group inner_schema {
              REQUIRED INT32 b;
              REQUIRED DOUBLE c;
            }

            OPTIONAL group e (LIST) {
              REPEATED group list {
                REQUIRED INT32 element;
              }
            }
        ";

        let parquet_type = Rc::new(parse_message_type(&message_type).unwrap());

        let mut visitor = TestVisitor::new(parquet_type.clone());
        for f in parquet_type.get_fields() {
            let c = TestVisitorContext {};
            assert!(visitor.dispatch(f.clone(), c).unwrap());
        }

        assert!(visitor.struct_visited);
        assert!(visitor.primitive_visited);
        assert!(visitor.list_visited);
    }
}
