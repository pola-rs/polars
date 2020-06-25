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

use std::process::Command;

fn main() {
    // Set Parquet version, build hash and "created by" string.
    let version = env!("CARGO_PKG_VERSION");
    let mut created_by = format!("parquet-rs version {}", version);
    if let Ok(git_hash) = run(Command::new("git").arg("rev-parse").arg("HEAD")) {
        created_by.push_str(format!(" (build {})", git_hash).as_str());
        println!("cargo:rustc-env=PARQUET_BUILD={}", git_hash);
    }
    println!("cargo:rustc-env=PARQUET_VERSION={}", version);
    println!("cargo:rustc-env=PARQUET_CREATED_BY={}", created_by);
}

/// Runs command and returns either content of stdout for successful execution,
/// or an error message otherwise.
fn run(command: &mut Command) -> Result<String, String> {
    println!("Running: `{:?}`", command);
    match command.output() {
        Ok(ref output) if output.status.success() => {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        }
        Ok(ref output) => Err(format!("Failed: `{:?}` ({})", command, output.status)),
        Err(error) => Err(format!("Failed: `{:?}` ({})", command, error)),
    }
}
