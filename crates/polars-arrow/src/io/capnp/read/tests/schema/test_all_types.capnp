# Copyright (c) 2013-2014 Sandstorm Development Group, Inc. and contributors
# Licensed under the MIT License:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

@0xd508eebdc2dc42b8;

enum TestEnum {
  foo @0;
  bar @1;
  baz @2;
  qux @3;
  quux @4;
  corge @5;
  grault @6;
  garply @7;
}

struct TestAllTypes {
  voidField      @0  : Void;
  boolField      @1  : Bool;
  int8Field      @2  : Int8;
  int16Field     @3  : Int16;
  int32Field     @4  : Int32;
  int64Field     @5  : Int64;
  uInt8Field     @6  : UInt8;
  uInt16Field    @7  : UInt16;
  uInt32Field    @8  : UInt32;
  uInt64Field    @9  : UInt64;
  float32Field   @10 : Float32;
  float64Field   @11 : Float64;
  textField      @12 : Text;
  dataField      @13 : Data;
  structField    @14 : TestAllTypes;
  enumField      @15 : TestEnum;

  voidList      @16 : List(Void);
  boolList      @17 : List(Bool);
  int8List      @18 : List(Int8);
  int16List     @19 : List(Int16);
  int32List     @20 : List(Int32);
  int64List     @21 : List(Int64);
  uInt8List     @22 : List(UInt8);
  uInt16List    @23 : List(UInt16);
  uInt32List    @24 : List(UInt32);
  uInt64List    @25 : List(UInt64);
  float32List   @26 : List(Float32);
  float64List   @27 : List(Float64);
  textList      @28 : List(Text);
  dataList      @29 : List(Data);
  structList    @30 : List(TestAllTypes);
  enumList      @31 : List(TestEnum);
}

struct TestEmptyStruct {
  struct EmptyStruct {}
  emptyField @0 : EmptyStruct;
}

struct TestUnion {
  union0 :union {
    foo @0 :UInt16;
    bar @1 :UInt32;
  }

  struct OuterStruct {
    struct InnerStruct {
      baz @0 :UInt16;
    }
  
    union1 :union {
      listInner @0 :List(InnerStruct);
      qux @1 :UInt32;
    }
    union {
      primitiveList @2 :List(UInt16);
      corge @3 :InnerStruct;
    }
  }
  listOuter @2 :List(OuterStruct);

  union {
    grault @3 :UInt16;
    garply @4 :Text;
  }
}
