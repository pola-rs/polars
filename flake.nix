{
  description = "A basic Nix Flake for eachDefaultSystem";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
    utils.url = "github:numtide/flake-utils";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "utils";
      };
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      utils,
      pyproject-nix,
      rust-overlay,
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            rust-overlay.overlays.default
          ];
        };
        lib = pkgs.lib;

        rustNightlyToolchain = (pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml).override {
          extensions = [
            "rust-analyzer"
            "rust-src"
            "llvm-tools"
            "miri"
          ];
          targets = [ "wasm32-unknown-emscripten" ];
        };

        # The extra-index-url produces errors that we cannot really deal with.
        requirements-ci-text =
          builtins.replaceStrings [ "--extra-index-url https://download.pytorch.org/whl/cpu" ] [ "" ]
            (builtins.readFile ./py-polars/requirements-ci.txt);

        requirements-dev = pyproject-nix.lib.project.loadRequirementsTxt {
          requirements = builtins.readFile ./py-polars/requirements-dev.txt;
        };
        requirements-lint = pyproject-nix.lib.project.loadRequirementsTxt {
          requirements = builtins.readFile ./py-polars/requirements-lint.txt;
        };
        requirements-ci = pyproject-nix.lib.project.loadRequirementsTxt {
          requirements = requirements-ci-text;
        };
        requirements-docs = pyproject-nix.lib.project.loadRequirementsTxt {
          requirements = builtins.readFile ./py-polars/docs/requirements-docs.txt;
        };

        packageOverrides = self: super: rec {
          maturin = [ ];
          patchelf = [ ];
          polars-cloud = [ ];

          fastexcel = super.buildPythonPackage rec {
            pname = "fastexcel";
            version = "0.13.0";
            pyproject = true;

            src = pkgs.fetchFromGitHub {
              owner = "ToucanToco";
              repo = "fastexcel";
              tag = "v${version}";
              hash = "sha256-o2+LNpl431/l4YL5/jnviDwZ5D+WjcFRoNV5hLuvRhM=";
            };

            cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
              inherit pname version src;
              hash = "sha256-VZoloGsYLAHqeqRkeZi0PZUpN/i+bWlebzL4wDZNHeo=";
            };

            nativeBuildInputs = with pkgs; [
              cargo
              rustPlatform.cargoSetupHook
              rustPlatform.maturinBuildHook
              rustc
            ];

            dependencies = [
              super.pyarrow
            ];

            optional-dependencies = {
              pandas = [
                super.pandas
              ];
              polars = [
                super.polars
              ];
            };

            pythonImportsCheck = [
              "fastexcel"
            ];
          };
          pyiceberg = super.buildPythonPackage rec {
            pname = "iceberg-python";
            version = "0.9.0";
            pyproject = true;

            src = pkgs.fetchFromGitHub {
              owner = "apache";
              repo = "iceberg-python";
              tag = "pyiceberg-${version}";
              hash = "sha256-PLxYe6MpKR6qILTNt0arujyx/nlVorwjhwokbXvdwb0=";
            };

            patches = [
              # Build script fails to build the cython extension on python 3.11 (no issues with python 3.12):
              # distutils.errors.DistutilsSetupError: each element of 'ext_modules' option must be an Extension instance or 2-tuple
              # This error vanishes if Cython and setuptools imports are swapped
              # https://stackoverflow.com/a/53356077/11196710
              (pkgs.writeText "test" ''
                diff --git a/build-module.py b/build-module.py
                index d91375e..4d307e8 100644
                --- a/build-module.py
                +++ b/build-module.py
                @@ -23,10 +23,10 @@ allowed_to_fail = os.environ.get("CIBUILDWHEEL", "0") != "1"
                 
                 
                 def build_cython_extensions() -> None:
                -    import Cython.Compiler.Options
                -    from Cython.Build import build_ext, cythonize
                     from setuptools import Extension
                     from setuptools.dist import Distribution
                +    import Cython.Compiler.Options
                +    from Cython.Build import build_ext, cythonize
                 
                     Cython.Compiler.Options.annotate = True
                							'')
            ];

            build-system = with super; [
              cython
              poetry-core
              setuptools
            ];

            # Prevents the cython build to fail silently
            env.CIBUILDWHEEL = "1";

            dependencies = with super; [
              cachetools
              click
              fsspec
              mmh3
              pydantic
              pyparsing
              ray
              requests
              rich
              sortedcontainers
              strictyaml
              tenacity
              zstandard
            ];
          };
          pytest-codspeed = super.buildPythonPackage rec {
            pname = "pytest-codspeed";
            version = "3.2.0";
            pyproject = true;

            src = pkgs.fetchFromGitHub {
              owner = "CodSpeedHQ";
              repo = "pytest-codspeed";
              tag = "v${version}";
              hash = "sha256-SNVJtnanaSQTSeX3EFG+21GFC1WFCQTbaNyi7QjQROw=";
            };

            build-system = [ super.hatchling ];

            buildInputs = [ super.pytest ];

            dependencies = with super; [
              cffi
              filelock
              importlib-metadata
              rich
              setuptools
            ];

            optional-dependencies = with super; {
              compat = [
                pytest-benchmark
                pytest-xdist
              ];
            };

            nativeCheckInputs = with super; [
              semver
              pytest-cov-stub
              pytestCheckHook
            ];

            pythonImportsCheck = [ "pytest_codspeed" ];
          };
          sphinx-favicon = super.buildPythonPackage rec {
            pname = "sphinx-favicon";
            version = "1.0.1";
            pyproject = true;

            src = pkgs.fetchFromGitHub {
              owner = "tcmetzger";
              repo = "sphinx-favicon";
              tag = "v${version}";
              hash = "sha256-Arcjj+6WWuSfufh8oqrDyAtjp07j1JEuw2YlmFcfL3U=";
            };

            build-system = with super; [
              setuptools
            ];

            dependencies = with super; [
              sphinx
            ];

            nativeCheckInputs = with super; [
              pytestCheckHook
            ];

            checkInputs = with super; [
              beautifulsoup4
            ];

            disabledTests = [
              # requires network to download favicons
              "test_list_of_three_icons_automated_values"
            ];

            pythonImportsCheck = [ "sphinx_favicon" ];
          };
          sphinx-reredirects = super.buildPythonPackage rec {
            pname = "sphinx-reredirects";
            version = "0.1.6";
            pyproject = true;

            src = pkgs.fetchPypi {
              pname = "sphinx_reredirects";
              inherit version;
              hash = "sha256-xJHLpUX2e+lpdQhyeBjYYmYmNmJFrmRFb+KfN+m76mQ=";
            };

            build-system = with super; [
              setuptools
            ];

            dependencies = with super; [
              sphinx
            ];

            pythonImportsCheck = [
              "sphinx_reredirects"
            ];
          };
          dunamai = super.dunamai.overridePythonAttrs (old: {
            doCheck = false;
          });
          adbc-driver-manager = super.buildPythonPackage rec {
            pname = "adbc_driver_manager";
            version = "0.11.0";
            format = "wheel";

            doCheck = false;
            src = super.fetchPypi {
              inherit pname version format;
              dist = "cp311";
              python = "cp311";
              abi = "cp311";
              platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
              sha256 = "sha256-bhWC60UyupxcDhPH3sYQAm3G3IPAUM7pszIJ1o8IsF4=";
            };
          };
          adbc-driver-sqlite = super.buildPythonPackage rec {
            pname = "adbc_driver_sqlite";
            version = "0.11.0";
            format = "wheel";

            doCheck = false;
            src = super.fetchPypi {
              inherit pname version format;
              dist = "py3";
              python = "py3";
              abi = "none";
              platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
              sha256 = "sha256-bazbckm+VAoe3TatTzxtKnwbbzdCx4Yjnr5gfuXPKYs=";
            };
          };
          connectorx = super.buildPythonPackage rec {
            pname = "connectorx";
            version = "0.4.3";
            format = "pyproject";

            src = pkgs.fetchFromGitHub {
              owner = "sfu-db";
              repo = "connector-x";
              rev = "v${version}";
              hash = "sha256-+2lXwxehqeCqD/R1AkCUrioX/wZDm2QZ35RMptqpqzs=";
            };

            sourceRoot = "${src.name}/connectorx-python";

            cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
              inherit src sourceRoot;
              name = "${pname}-python-${version}";
              hash = "sha256-vBFaUmXkKGC6DaB5Ee/cbXDh3tO04NxTx4UPfWJoRvA=";
            };

            env = {
              # needed for openssl-sys
              OPENSSL_NO_VENDOR = 1;
              OPENSSL_LIB_DIR = "${pkgs.lib.getLib pkgs.openssl}/lib";
              OPENSSL_DIR = "${pkgs.lib.getDev pkgs.openssl}";
            };

            nativeBuildInputs = [
              pkgs.krb5 # needed for `krb5-config` during libgssapi-sys

              pkgs.rustPlatform.cargoSetupHook
              pkgs.rustPlatform.maturinBuildHook
              pkgs.rustPlatform.bindgenHook
            ];

            # nativeCheckInputs = [ pytestCheckHook ];

            buildInputs = with pkgs; [
              libkrb5 # needed for libgssapi-sys
              openssl # needed for openssl-sys
            ];

            pythonImportsCheck = [ "connectorx" ];

          };
          kuzu = super.buildPythonPackage rec {
            pname = "kuzu";
            version = "0.4.1";
            format = "wheel";

            doCheck = false;
            src = super.fetchPypi {
              inherit pname version format;
              python = "cp311";
              dist = "cp311";
              abi = "cp311";
              platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
              sha256 = "sha256-rToMGPn2vCbk+EinLZ8yvYsHldVK8UcFFlss0Q9CgPs=";
            };
          };
          google-auth-stubs =
            let
              grpc-stubs = super.buildPythonPackage rec {
                pname = "grpc-stubs";
                version = "1.53.0.5";
                format = "pyproject";

                # disabled = pythonVersion < 3.7

                dependencies = with super; [
                  pytest
                  cffi
                  filelock
                  hatchling
                  setuptools
                ];

                nativeBuildInputs = with super; [
                  poetry-core
                ];

                propagatedBuildInputs = with super; [
                  grpcio
                ];

                src = pkgs.fetchFromGitHub {
                  owner = "shabbyrobe";
                  repo = "${pname}";
                  rev = "${version}";
                  hash = "sha256-an7xztaCqxOEmf74Rgb8q9u/WsojFYkBiwtLRa1qqBQ=";
                };
              };
            in
            super.buildPythonPackage rec {
              pname = "google-auth-stubs";
              version = "0.3.0";
              format = "pyproject";

              # disabled = pythonVersion < 3.7

              dependencies = with super; [
                pytest
                cffi
                filelock
                hatchling
                setuptools
              ];

              nativeBuildInputs = with super; [
                poetry-core
              ];

              propagatedBuildInputs = with super; [
                google-auth
                grpc-stubs
                types-requests
              ];

              src = pkgs.fetchFromGitHub {
                owner = "henribru";
                repo = "${pname}";
                rev = "v${version}";
                hash = "sha256-xTJ+MaOZN7jgjSSKB36bcADXC28wUh22DAezZMVd+mk=";
              };
            };
          autodocsumm = super.buildPythonPackage rec {
            pname = "autodocsumm";
            version = "0.2.14";
            src = super.fetchPypi {
              inherit pname version;
              hash = "sha256-KDmp1PrMPE7M0wbAhpVUCREEK0bur83DID5tC6tAvHc=";
            };

            dependencies = with super; [
              sphinx
              versioneer
              setuptools
            ];
          };
          sphinx-autosummary-accessors = super.buildPythonPackage {
            pname = "sphinx-autosummary-accessors";
            version = "2023.4.0";
            pyproject = true;

            src = pkgs.fetchFromGitHub {
              owner = "xarray-contrib";
              repo = "sphinx-autosummary-accessors";
              rev = "2023.04.0";
              hash = "sha256-s0epnJLRwTVXn8Y4tzd2i9qkGSXPG2lTL4e0Q4z9eYo=";
            };

            dependencies = with super; [
              sphinx
              setuptools
              setuptools-scm
            ];

            pythonImportsCheck = [ "sphinx_autosummary_accessors" ];
          };
          sphinx-toolbox =
            let
              sphinx-jinja2-compat = super.buildPythonPackage rec {
                pname = "sphinx-jinja2-compat";
                version = "0.3.0";
                pyproject = true;

                dependencies = with super; [
                  whey
                  whey-pth
                  jinja2
                  markupsafe
                ];

                src = pkgs.fetchFromGitHub {
                  owner = "sphinx-toolbox";
                  repo = pname;
                  rev = "v${version}";
                  hash = "sha256-MsmeZP96Lrxvfx07yX1fDgXuxrsjx25uGGrIJYRWlbg=";
                };
              };
              dict2css = super.buildPythonPackage rec {
                pname = "dict2css";
                version = "0.3.0";
                pyproject = true;

                dependencies = with super; [
                  whey
                  cssutils
                ];

                src = pkgs.fetchFromGitHub {
                  owner = "sphinx-toolbox";
                  repo = "dict2css";
                  rev = "v${version}";
                  hash = "sha256-PkoXFSbTJaYfhb1ba6qUIQ3e9dYNpeTXmCLc39hhrF4=";
                };
              };
              apeye = super.buildPythonPackage rec {
                pname = "apeye";
                version = "1.4.1";
                pyproject = true;

                doCheck = false;
                dependencies = with super; [
                  flit-core
                  apeye-core
                  requests
                  platformdirs
                ];

                src = pkgs.fetchFromGitHub {
                  owner = "domdfcoding";
                  repo = "apeye";
                  rev = "v${version}";
                  hash = "sha256-kxFVsGMqOrrelqiiRh7U/VdG/1WTY6MxCKI/keUjBTM=";
                };
              };
            in
            super.buildPythonPackage rec {
              pname = "sphinx-toolbox";
              version = "3.5.0";
              pyproject = true;

              src = pkgs.fetchFromGitHub {
                owner = "sphinx-toolbox";
                repo = "sphinx-toolbox";
                rev = "v${version}";
                hash = "sha256-UjXWj5jrgDDuyljsf0XCnvigV4BpZ02Wb6QxN+sXDXs=";
              };

              doCheck = false;
              dependencies = with super; [
                sphinx
                whey
                setuptools
                dict2css
                autodocsumm
                apeye
                beautifulsoup4
                cachecontrol
                sphinx-tabs
                sphinx-prompt
                sphinx-autodoc-typehints
                sphinx-jinja2-compat
                filelock
                html5lib
                ruamel-yaml
                tabulate
              ];

              # These are some hacks to ensure that we don't create a cache at
              # root.
              prePatch = ''
                							substituteInPlace sphinx_toolbox/__init__.py      --replace 'from sphinx_toolbox.cache import cache' "# removed"
                							substituteInPlace sphinx_toolbox/github/issues.py --replace 'from sphinx_toolbox.cache import cache' "# removed"
                							rm -rf sphinx_toolbox/cache.py

                							cat sphinx_toolbox/__init__.py
                						'';

              pythonImportsCheck = [ "sphinx_toolbox" ];
            };
          typos = [ ];
        };

        python = pkgs.python311.override {
          inherit packageOverrides;
        };

        pythonEnv =
          # Assert that versions from nixpkgs matches what's described in requirements.txt
          # In projects that are overly strict about pinning it might be best to remove this assertion entirely.
          # assert requirements-dev.validators.validateVersionConstraints { inherit python; } == { };
          # assert requirements-lint.validators.validateVersionConstraints { inherit python; } == { };
          # assert requirements-ci.validators.validateVersionConstraints { inherit python; } == { };
          # assert requirements-docs.validators.validateVersionConstraints { inherit python; } == { };
          (
            # Render requirements.txt into a Python withPackages environment
            python.withPackages (
              python-pkgs:
              let
                packages-dev = (requirements-dev.renderers.withPackages { inherit python; }) python-pkgs;
                packages-ci = (requirements-ci.renderers.withPackages { inherit python; }) python-pkgs;
                packages-docs = (requirements-docs.renderers.withPackages { inherit python; }) python-pkgs;
                packages-lint = (requirements-lint.renderers.withPackages { inherit python; }) python-pkgs;
              in
              (
                packages-dev
                ++ packages-ci
                ++ packages-docs
                ++ packages-lint
                ++ (with python-pkgs; [
                  importlib-resources
                  psutil
                  hvplot
                  seaborn

                  duckdb
                  pandas
                  jupyterlab

                  pygithub

                  # Used for polars-benchmark
                  pydantic-settings

                  # # Used for Altair SVG / PNG conversions
                  # (localPyPkg ./python-packages/vl-convert-python)
                ])
              )
            )
          );
      in
      {
        devShells.default =
          let
            aliasToScript =
              alias:
              let
                pwd = if alias ? pwd then "$POLARS_ROOT/${alias.pwd}" else "$POLARS_ROOT";
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
                targetDir = "$POLARS_ROOT/py-polars/polars";
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

                                  ln -sf "$POLARS_ROOT/py-polars/polars/polars.abi3.so.$1.latest" polars/polars.abi3.so
                '';
                doc = "Build the python library";
              };
              pybuild = {
                pwd = "py-polars";
                cmd = buildPy "debug" "maturin develop -m $POLARS_ROOT/py-polars/Cargo.toml \"$@\"";
                doc = "Build the python library";
              };
              pybuild-mindebug = {
                pwd = "py-polars";
                cmd = buildPy "mindebug" "maturin develop --profile mindebug-dev \"$@\"";
                doc = "Build the python library with minimal debug information";
              };
              pybuild-nodebug-release = {
                pwd = "py-polars";
                cmd = buildPy "nodebug-release" "maturin develop --profile nodebug-release \"$@\"";
                doc = "Build the python library in release mode without debug symbols";
              };
              pybuild-release = {
                pwd = "py-polars";
                cmd = buildPy "release" "maturin develop --profile release \"$@\"";
                doc = "Build the python library in release mode with minimal debug symbols";
              };
              pybuild-debug-release = {
                pwd = "py-polars";
                cmd = buildPy "debug-release" "maturin develop --profile debug-release \"$@\"";
                doc = "Build the python library in release mode with full debug symbols";
              };
              pybuild-dist-release = {
                pwd = "py-polars";
                cmd = buildPy "dist-release" "maturin develop --profile dist-release \"$@\"";
                doc = "Build the python library in release mode which would be distributed to users";
              };
              pyselect-build = {
                pwd = "py-polars";
                cmd = ''
                  if [ -z "$1" ]; then
                      echo "Usage: $0 <BUILD>" > 2
                      exit 2
                  fi

                  TARGET_DIR="$POLARS_ROOT/py-polars/polars"
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
          in
          pkgs.mkShell {
            packages =
              with pkgs;
              [
                pythonEnv

                cmake
                gnumake

                maturin
                rustNightlyToolchain

                typos
                mypy
                dprint

                zlib

                cargo-nextest

                linuxPackages_latest.perf
                samply
                hyperfine

                graphviz

                openssl
                pkg-config
              ]
              ++ (mapAttrsToList (
                name: value: pkgs.writeShellScriptBin "pl-${name}" (aliasToScript value)
              ) aliases);

            shellHook =
              let
                concatStrings = lib.concatStrings;

                max =
                  x: y:
                  assert builtins.isInt x;
                  assert builtins.isInt y;
                  if x < y then y else x;
                listMax = lib.foldr max 0;
                maxLength = listMax (mapAttrsToList (name: _: (builtins.stringLength name)) aliases);
                nSpaces = n: (lib.concatMapStrings (_: " ") (lib.range 1 n));
              in
              ''
                						export POLARS_ROOT="$PWD"
                            export PYTHONPATH="$PYTHONPATH:$PWD/py-polars"
                            export CARGO_BUILD_JOBS=8

                            echo
                            echo 'Defined Aliases:'
                            ${concatStrings (
                              mapAttrsToList (name: value: ''
                                echo ' - pl-${name}:${nSpaces (maxLength - (builtins.stringLength name))} ${value.doc}'
                              '') aliases
                            )}
              '';

          };
        packages.default =
          let
            project = builtins.fromTOML (builtins.readFile ./py-polars/Cargo.toml);
            rustNightlyPlatform = pkgs.makeRustPlatform {
              cargo = rustNightlyToolchain;
              rustc = rustNightlyToolchain;
            };
          in
          pkgs.python3Packages.buildPythonPackage {
            pname = "polars";
            version = project.package.version;

            build-system = [ rustNightlyPlatform.maturinBuildHook ];

            nativeBuildInputs = with pkgs; [
              pkg-config
              rustNightlyPlatform.cargoSetupHook
              rustNightlyPlatform.cargoBuildHook
              rustNightlyPlatform.cargoInstallHook
              rustNightlyToolchain
            ];

            maturinBuildFlags = [
              "-m"
              "py-polars/Cargo.toml"
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
            };
          };
      }
    );
}
