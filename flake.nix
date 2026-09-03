{
  description = "Development environment for Polars";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
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
          rustToolchain = (pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml).override {
            extensions = [
              "rust-analyzer"
              "rust-src"
            ];
          };

          # Create an alias for python packages, such that we can use the same python version for everything
          py = pkgs.python313Packages;
        in
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ inputs.rust-overlay.overlays.default ];
          };

          devShells.default = pkgs.mkShell {
            packages =
              (with pkgs; [
                py.python
                rustToolchain
                cargo-nextest
                cmake
                dprint
                gnumake
                graphviz
                hyperfine
                pkg-config
                samply
              ])
              ++ (lib.optional pkgs.stdenv.hostPlatform.isLinux pkgs.perf);

            buildInputs = with pkgs; [ openssl ];

            shellHook =
              let
                openCmd = if pkgs.stdenv.hostPlatform.isLinux then "xdg-open" else "open";
              in
              ''
                export WORKSPACE_ROOT=$(git rev-parse --show-toplevel)

                # Jemmalloc compiled with gcc doesn't like when we ask for the
                # compiler to compile with fortify source so lets enable everything
                # but fortify and fortify3.
                export NIX_HARDENING_ENABLE="bindnow format pic relro stackclashprotection stackprotector strictoverflow zerocallusedregs"

                export PYO3_NO_RECOMPILE=1

                # - cc is needed for numpy to function
                # - python shared libs are required for rust-side tests
                export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${py.python}/lib"

                export POLARS_DOT_SVG_VIEWER="${openCmd} %file%"
                export RUST_SRC_PATH="${rustToolchain}/lib/rustlib/src/rust/library"

                # Create the virtual environment and install the Python
                # requirements if they are missing; a no-op otherwise.
                make -s -C "$WORKSPACE_ROOT" .venv
                source "$WORKSPACE_ROOT/.venv/bin/activate"
              '';
          };
        };
    };
}
