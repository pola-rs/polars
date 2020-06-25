#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Change to the toplevel Rust directory
pushd $DIR/../../

# Execute the code generation:
flatc --rust -o rust/arrow/src/ipc/gen/ format/*.fbs

# Now the files are wrongly named so we have to change that.
popd
pushd $DIR/src/ipc/gen
for f in `ls *_generated.rs`; do
    adj_length=$((${#f}-13))
    mv $f "${f:0:$adj_length}.rs"
done

PREFIX=$(cat <<'HEREDOC'
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

#![allow(dead_code)]
#![allow(unused_imports)]

use std::{cmp::Ordering, mem};
use flatbuffers::EndianScalar;

HEREDOC
)

SCHEMA_IMPORT="\nuse crate::ipc::gen::Schema::*;"
SPARSE_TENSOR_IMPORT="\nuse crate::ipc::gen::SparseTensor::*;"
TENSOR_IMPORT="\nuse crate::ipc::gen::Tensor::*;"


# Remove all generated lines we don't need
for f in `ls *.rs`; do

    if [[ $f == "mod.rs" ]]; then
        continue
    fi

    echo "Modifying: $f"
    sed -i '' '/extern crate flatbuffers;/d' $f
    sed -i '' '/use self::flatbuffers::EndianScalar;/d' $f
    sed -i '' '/\#\[allow(unused_imports, dead_code)\]/d' $f
    sed -i '' '/pub mod org {/d' $f
    sed -i '' '/pub mod apache {/d' $f
    sed -i '' '/pub mod arrow {/d' $f
    sed -i '' '/pub mod flatbuf {/d' $f
    sed -i '' '/}  \/\/ pub mod flatbuf/d' $f
    sed -i '' '/}  \/\/ pub mod arrow/d' $f
    sed -i '' '/}  \/\/ pub mod apache/d' $f
    sed -i '' '/}  \/\/ pub mod org/d' $f
    sed -i '' '/use std::mem;/d' $f
    sed -i '' '/use std::cmp::Ordering;/d' $f

    # Replace all occurrences of type__ with type_
    sed -i '' 's/type__/type_/g' $f

    # Some files need prefixes
    if [[ $f == "File.rs" ]]; then 
        # Now prefix the file with the static contents
        echo -e "${PREFIX}" "${SCHEMA_IMPORT}" | cat - $f > temp && mv temp $f
    elif [[ $f == "Message.rs" ]]; then
        echo -e "${PREFIX}" "${SCHEMA_IMPORT}" "${SPARSE_TENSOR_IMPORT}" "${TENSOR_IMPORT}" | cat - $f > temp && mv temp $f
    elif [[ $f == "SparseTensor.rs" ]]; then
        echo -e "${PREFIX}" "${SCHEMA_IMPORT}" "${TENSOR_IMPORT}" | cat - $f > temp && mv temp $f
    elif [[ $f == "Tensor.rs" ]]; then
        echo -e "${PREFIX}" "${SCHEMA_IMPORT}" | cat - $f > temp && mv temp $f
    else
        echo "${PREFIX}" | cat - $f > temp && mv temp $f
    fi
done

# Return back to base directory
popd
cargo +stable fmt -- src/ipc/gen/*