{
  description = "A basic Nix Flake for eachDefaultSystem";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } (
      { ... }:
      {
        systems = [
          "x86_64-linux"
          "aarch64-linux"
          "x86_64-darwin"
          "aarch64-darwin"
        ];
        perSystem =
          {
            system,
            pkgs,
            lib,
            self',
            ...
          }:
          let
            rustToolchain = pkgs.fenix.fromToolchainName {
              name = (lib.importTOML ./rust-toolchain.toml).toolchain.channel;
              sha256 = "sha256-xJcQgGnbraFO5NipYwkHR+V1BxGbTe1ZrRnNw5InBEg=";
            };

            rustPlatform = pkgs.makeRustPlatform {
              cargo = rustToolchain;
              rustc = rustToolchain;
            };

            # Create an alias for python packages, such that we can use the same python version for everything
            py = pkgs.python313Packages;

          in
          {
            _module.args.pkgs = import inputs.nixpkgs {
              inherit system;
              overlays = [ inputs.fenix.overlays.default ];
              config = { };
            };

            # packages.polars =
            #   let
            #     project = builtins.fromTOML (builtins.readFile ./py-polars/runtime/polars-runtime-32/Cargo.toml);
            #   in
            #   pythonPlatform.buildPythonPackage {
            #     pname = "polars";
            #     version = project.package.version;
            #
            #     build-system = [ rustPlatform.maturinBuildHook ];
            #
            #     nativeBuildInputs = [
            #       rustPlatform.cargoSetupHook
            #       rustPlatform.cargoBuildHook
            #       rustPlatform.cargoInstallHook
            #     ];
            #
            #     src = ./.;
            #
            #     cargoRoot = "client";
            #
            #     maturinBuildFlags = [
            #       "-m"
            #       "py-polars/runtime/polars-runtime-32/Cargo.toml"
            #       "--uv"
            #     ];
            #
            #     cargoDeps = rustPlatform.importCargoLock {
            #       lockFile = ./Cargo.lock;
            #     };
            #   };

            devShells.default = pkgs.mkShell (
              let
                runtimePkgs =
                  with pkgs;
                  lib.optionals stdenv.isLinux [
                    gcc13
                    openssl_3_6
                  ];
                stdenv = pkgs.stdenv;

                aliasToScript =
                  alias:
                  let
                    pwd = if alias ? pwd then "$WORKSPACE_ROOT/${alias.pwd}" else "$WORKSPACE_ROOT";
                  in
                  ''
                    set -e
                    pushd "${pwd}" > /dev/null
                    echo "[INFO]: Changed directory to ${pwd}"
                    ${alias.cmd}
                    popd > /dev/null
                  '';
                buildPy =
                  alias: cmd:
                  let
                    targetDir = "$WORKSPACE_ROOT/py-polars/polars";
                  in
                  ''
                    ${cmd}
                    mv "${targetDir}/polars.abi3.so" "${targetDir}/polars.abi3.so.${alias}"
                    ln -sf "${targetDir}/polars.abi3.so.${alias}" "${targetDir}/polars.abi3.so"
                  '';
                step = title: alias: ''
                  echo '[${title}]'
                  ${aliasToScript alias}
                  echo '${title} Done ✅'
                  echo
                '';
                aliases = rec {
                  check = {
                    cmd = "cargo check --workspace --all-targets --all-features";
                    doc = "Run cargo check with all features";
                  };
                  typos = {
                    cmd = "typos";
                    doc = "Run a Spell Check with Typos";
                  };
                  clippy-all = {
                    cmd = "cargo clippy --workspace --all-targets --all-features --locked -- -D warnings -D clippy::dbg_macro";
                    doc = "Run clippy with all features";
                  };
                  clippy-default = {
                    cmd = "cargo clippy --all-targets --locked -- -D warnings -D clippy::dbg_macro";
                    doc = "Run clippy with default features";
                  };
                  fmt = {
                    cmd = "cargo fmt --all";
                    doc = "Run autoformatting";
                  };
                  pyselect = {
                    pwd = "py-polars";
                    cmd = ''
                      if [ -z "$1" ]; then
                        echo "Usage $0 <debug/debug-release>"
                        exit 2
                      fi

                      ln -sf "$WORKSPACE_ROOT/py-polars/polars/polars.abi3.so.$1.latest" polars/polars.abi3.so
                    '';
                    doc = "Build the python library";
                  };
                  pybuild = {
                    pwd = "py-polars";
                    cmd = buildPy "debug" "maturin develop -m $WORKSPACE_ROOT/py-polars/runtime/polars-runtime-32/Cargo.toml \"$@\" --uv";
                    doc = "Build the python library";
                  };
                  pybuild-mindebug = {
                    pwd = "py-polars";
                    cmd = buildPy "mindebug" "maturin develop --profile mindebug-dev \"$@\" --uv";
                    doc = "Build the python library with minimal debug information";
                  };
                  pybuild-nodebug-release = {
                    pwd = "py-polars";
                    cmd = buildPy "nodebug-release" "maturin develop --profile nodebug-release \"$@\" --uv";
                    doc = "Build the python library in release mode without debug symbols";
                  };
                  pybuild-release = {
                    pwd = "py-polars";
                    cmd = buildPy "release" "maturin develop --profile release \"$@\" --uv";
                    doc = "Build the python library in release mode with minimal debug symbols";
                  };
                  pybuild-debug-release = {
                    pwd = "py-polars";
                    cmd = buildPy "debug-release" "maturin develop --profile debug-release \"$@\" --uv";
                    doc = "Build the python library in release mode with full debug symbols";
                  };
                  pybuild-dist-release = {
                    pwd = "py-polars";
                    cmd = buildPy "dist-release" "maturin develop --profile dist-release \"$@\" --uv";
                    doc = "Build the python library in release mode which would be distributed to users";
                  };
                  pyselect-build = {
                    pwd = "py-polars";
                    cmd = ''
                      if [ -z "$1" ]; then
                          echo "Usage: $0 <BUILD>" > 2
                          exit 2
                      fi

                      TARGET_DIR="$WORKSPACE_ROOT/py-polars/polars"
                      ln -sf "$TARGET_DIR/polars.abi3.so.$1" "$TARGET_DIR/polars.abi3.so"
                    '';
                    doc = "Select a previous build of polars";
                  };
                  pytest-all = {
                    pwd = "py-polars";
                    cmd = "pytest -n auto --dist=loadgroup \"$@\"";
                    doc = "Run the default python tests";
                  };
                  pytest-release = {
                    pwd = "py-polars";
                    cmd = "pytest -n auto --dist=loadgroup -m 'not release and not benchmark and not docs' \"$@\"";
                    doc = "Run the release python tests";
                  };
                  pytest = {
                    pwd = "py-polars";
                    cmd = "pytest \"$@\"";
                    doc = "Run the default python tests";
                  };
                  pyfmt = {
                    pwd = "py-polars";
                    cmd = ''
                      ruff check py-polars
                      ruff format py-polars
                      dprint fmt crates
                      typos crates
                      typos py-polars
                    '';
                    doc = "Run python autoformatting";
                  };
                  rstest = {
                    pwd = "crates";
                    cmd = ''
                      cargo test --all-features \
                        -p polars-compute       \
                        -p polars-core          \
                        -p polars-io            \
                        -p polars-lazy          \
                        -p polars-ops           \
                        -p polars-plan          \
                        -p polars-row           \
                        -p polars-sql           \
                        -p polars-time          \
                        -p polars-utils         \
                        --                      \
                        --test-threads=2        \
                    '';
                    doc = "Run the Rust tests";
                  };
                  rsnextest = {
                    pwd = "crates";
                    cmd = ''
                      cargo nextest run --all-features \
                        -p polars-compute              \
                        -p polars-core                 \
                        -p polars-io                   \
                        -p polars-lazy                 \
                        -p polars-ops                  \
                        -p polars-plan                 \
                        -p polars-row                  \
                        -p polars-sql                  \
                        -p polars-time                 \
                        -p polars-utils                \
                    '';
                    doc = "Run the Rust tests with Cargo-Nextest";
                  };
                  precommit = {
                    cmd = ''
                      ${step "Rust Format" fmt}
                      ${step "Python Format" pyfmt}
                      ${step "Spell Check" typos}
                      ${step "Clippy All" clippy-all}
                      ${step "Clippy Default" clippy-default}
                    '';
                    doc = "Run the checks to do before committing";
                  };
                  prepush = {
                    cmd = ''
                      ${aliasToScript precommit}
                      ${step "Rust Tests" rstest}
                      ${step "Python Build" pybuild-mindebug}
                      ${step "Python Tests" pytest-all}
                    '';
                    doc = "Run the checks to do before pushing";
                  };
                  profile-setup = {
                    cmd = ''
                      echo '1'    | sudo tee /proc/sys/kernel/perf_event_paranoid
                      echo '1024' | sudo tee /proc/sys/kernel/perf_event_mlock_kb
                    '';
                    doc = "Setup the environment for profiling";
                  };
                  debug-setup = {
                    cmd = ''
                      echo '0' | sudo tee /proc/sys/kernel/yama/ptrace_scope
                    '';
                    doc = "Setup the environment for attach debugging";
                  };
                };

                mapAttrsToList = lib.attrsets.mapAttrsToList;
								extraPyDeps = [
									"importlib-resources"
									"psutil"
									"hvplot"
									"seaborn"

									"duckdb"
									"pandas"
									"jax"
									"torch"
									"jupyterlab"
									"pyiceberg"

									"pygithub"

									# Used for polars-benchmark
									"pydantic-settings"
									"ruff"

									# # Used for Altair SVG / PNG conversions
									"vl-convert-python"
								];

                rustPkg = rustToolchain.withComponents [
                  "cargo"
                  "clippy"
                  "rust-src"
                  "rustc"
                  "rustfmt"
                  "rust-analyzer"
                ];
              in
              {
                packages =
                  with pkgs;
                  [
                    py.python
                    py.venvShellHook
                    py.build
                    py.mypy

                    rustPkg

                    cmake
                    gnumake

                    maturin

                    typos
                    dprint
										uv

                    zlib

                    cargo-nextest

                    samply
                    hyperfine

                    graphviz

                    openssl
                    pkg-config
                  ]
                  ++ (mapAttrsToList (
                    name: value: pkgs.writeShellScriptBin "pl-${name}" (aliasToScript value)
                  ) aliases)
                  ++ (pkgs.lib.optionals pkgs.stdenv.isLinux [
                    pkgs.perf
                    mold-wrapped
                  ]);

                buildInputs = runtimePkgs;

                postVenvCreation = ''
                  unset CONDA_PREFIX 
                  MATURIN_PEP517_ARGS="--profile dev" uv pip install --upgrade --compile-bytecode --no-build \
                    -r py-polars/requirements-dev.txt \
                    -r py-polars/requirements-lint.txt \
                    -r py-polars/docs/requirements-docs.txt \
                    -r docs/source/requirements.txt \
                    ${builtins.concatStringsSep " " extraPyDeps} \
                  && uv pip install --upgrade --compile-bytecode "pyiceberg>=0.7.1" pyiceberg-core \
                	&& uv pip install --no-deps -e py-polars \
                	&& uv pip uninstall polars-runtime-compat polars-runtime-64  ## Uninstall runtimes which might take precedence over polars-runtime-32
                '';

                venvDir = ".venv";

                postShellHook =
                  let
                    openCmd = 
                      if pkgs.stdenv.isLinux then "xdg-open"
                      else "open";
                        # on darwin, /usr/bin/ld actually looks at the environment variable
                    # Borrowed from jujutsu's flake.nix
                    # on macOS and Linux, use faster parallel linkers that are much more
                    # efficient than the defaults. these noticeably improve link time even for
                    # medium sized rust projects like jj
                    rustLinkerFlags =
                      if pkgs.stdenv.isLinux then
                        [
                          "-fuse-ld=mold"
                          "-Wl,--compress-debug-sections=zstd"
                        ]
                      else if pkgs.stdenv.isDarwin then
                        # on darwin, /usr/bin/ld actually looks at the environment variable
                        # $DEVELOPER_DIR, which is set by the nix stdenv, and if set,
                        # automatically uses it to route the `ld` invocation to the binary
                        # within. in the devShell though, that isn't what we want; it's
                        # functional, but Xcode's linker as of ~v15 (not yet open source)
                        # is ultra-fast and very shiny; it is enabled via -ld_new, and on by
                        # default as of v16+
                        [
                          "--ld-path=$(unset DEVELOPER_DIR; /usr/bin/xcrun --find ld)"
                          "-ld_new"
                        ]
                      else
                        [ ];

                    rustLinkFlagsString = pkgs.lib.concatStringsSep " " (
                      pkgs.lib.concatMap (x: [
                        "-C"
                        "linker=clang"
                        "-C"
                        "link-arg=${x}"
                      ]) rustLinkerFlags
                    );
                  in
                  ''
                    export WORKSPACE_ROOT=$(git rev-parse --show-toplevel)
                    export VENV=$WORKSPACE_ROOT/.venv

                    # Jemmalloc compiled with gcc doesn't like when we ask for the
                    # compiler to compile with fortify source so lets enable everything
                    # but fortify and fortify3.
                    export NIX_HARDENING_ENABLE="bindnow format pic relro stackclashprotection stackprotector strictoverflow zerocallusedregs"

                    export PYO3_NO_RECOMPILE=1

                    export PYTHON_SHARED_LIB=$($VENV/bin/python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

                    # - cc is needed for numpy to function
                    # - python shared libs are required for rust-side tests
                    export LD_LIBRARY_PATH="${stdenv.cc.cc.lib}/lib:$PYTHON_SHARED_LIB"

                    export POLARS_DOT_SVG_VIEWER="${openCmd} %file%"
										export RUST_SRC_PATH="${rustToolchain.rust-src}/lib/rustlib/src/rust/library"
                  '';

              }
            );
            packages.polars =
              let
                project = builtins.fromTOML (builtins.readFile ./py-polars/runtime/polars-runtime-32/Cargo.toml);
              in
              py.buildPythonPackage {
                pname = "polars";
                version = project.package.version;

                build-system = [ rustToolchain.maturinBuildHook ];

                nativeBuildInputs = with pkgs; [
                  pkg-config
                  rustPlatform.cargoSetupHook
                  rustPlatform.cargoBuildHook
                  rustPlatform.cargoInstallHook
                  rustToolchain
                ];

                maturinBuildFlags = [
                  "-m"
                  "py-polars/runtime/polars-runtime-32/Cargo.toml"
                  "--uv"
                ];
                postInstall = ''
									# Move polars.abi3.so -> polars.so
									local polarsSo=""
									local soName=""
									while IFS= read -r -d "" p ; do
										polarsSo=$p
										soName="$(basename "$polarsSo")"
										[[ "$soName" == polars.so ]] && break
									done < <( find "$out" -iname "polars*.so" -print0 )
									[[ -z "''${polarsSo:-}" ]] && echo "polars.so not found" >&2 && exit 1
									if [[ "$soName" != polars.so ]] ; then
										mv "$polarsSo" "$(dirname "$polarsSo")/polars.so"
									fi
								'';

                src = ./.;
                cargoDeps = pkgs.rustPlatform.importCargoLock {
                  lockFile = ./Cargo.lock;
                  outputHashes = {
                    "pyo3-0.24.2" = "sha256-0V4cT3DstG9mZvdIVZXzoQlNyvtBuLOvlMe1XDZp3/0=";
                    "tikv-jemalloc-sys-0.6.0+5.3.0-1-ge13ca993e8ccb9ba9847cc330696e02839f328f7" =
                      "sha256-nvXKBd5tKSe4hPTtMKriYhlgAML9gdDHZG8nNRzgjXM=";
                  };
                };
              };
            packages.default = self'.packages.polars;
          };
      }
    );
}
