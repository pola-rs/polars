{
  description = "Development environment for Polars";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    systems.url = "github:nix-systems/triplet";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import inputs.systems;

      perSystem =
        {
          system,
          pkgs,
          lib,
          ...
        }:
        let
          rustToolchain = pkgs.fenix.fromToolchainName {
            name = (lib.importTOML ./rust-toolchain.toml).toolchain.channel;
            sha256 = "sha256-wBCNU5N9ftXKTMzvUW3xolIXmK5Z/93SdAxK1sMRDxQ=";
          };

          # Create an alias for python packages, such that we can use the same python version for everything
          py = pkgs.python313Packages;
        in
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ inputs.fenix.overlays.default ];
          };

          devShells.default = pkgs.mkShell (
            let
              runtimePkgs = lib.optionals pkgs.stdenv.hostPlatform.isLinux (
                with pkgs;
                [
                  gcc13
                  openssl_3_6
                ]
              );

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
                (with pkgs; [
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
                ])
                ++ (lib.optional pkgs.stdenv.hostPlatform.isLinux pkgs.perf);

              buildInputs = runtimePkgs;

              postVenvCreation = ''
                unset CONDA_PREFIX
                MATURIN_PEP517_ARGS="--profile dev" uv pip install --upgrade --compile-bytecode --no-build \
                  -r py-polars/requirements-dev.txt \
                  -r py-polars/requirements-lint.txt \
                  -r py-polars/docs/requirements-docs.txt \
                  -r docs/source/requirements.txt \
                  ${lib.join " " extraPyDeps} \
                && uv pip install --upgrade --compile-bytecode "pyiceberg>=0.7.1" pyiceberg-core \
                && uv pip install --no-deps -e py-polars \
                && uv pip uninstall polars-runtime-compat polars-runtime-64  ## Uninstall runtimes which might take precedence over polars-runtime-32
              '';

              venvDir = ".venv";

              postShellHook =
                let
                  openCmd = if pkgs.stdenv.hostPlatform.isLinux then "xdg-open" else "open";
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
                  export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$PYTHON_SHARED_LIB"

                  export POLARS_DOT_SVG_VIEWER="${openCmd} %file%"
                  export RUST_SRC_PATH="${rustToolchain.rust-src}/lib/rustlib/src/rust/library"
                '';
            }
          );
        };
    };
}
