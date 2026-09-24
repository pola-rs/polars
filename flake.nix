{
  description = "Development environment for Polars";

  inputs = {
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    systems.url = "github:nix-systems/triplet";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import inputs.systems;

      perSystem =
        {
          lib,
          pkgs,
          system,
          ...
        }:
        let
          python = pkgs.python313;

          rustToolchain = (pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml).override {
            extensions = [
              "rust-analyzer"
              "rust-src"
            ];
          };
        in
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ inputs.rust-overlay.overlays.default ];
          };

          checks.formatting =
            pkgs.runCommandLocal "check-formatting" { nativeBuildInputs = [ pkgs.nixfmt ]; }
              ''
                nixfmt --check ${./flake.nix}
                touch "$out"
              '';

          devShells.default = pkgs.mkShell {
            packages = [
              python
              rustToolchain
            ]
            ++ (with pkgs; [
              cargo-nextest
              cmake
              dprint
              git
              gnumake
              graphviz
              hyperfine
              pkg-config
              samply
            ])
            ++ lib.optional pkgs.stdenv.hostPlatform.isLinux pkgs.perf;

            buildInputs = with pkgs; [ openssl ];

            # jemalloc compiled with gcc doesn't like when we ask for the
            # compiler to compile with fortify source. Disabling fortify also
            # disables fortify3.
            hardeningDisable = [ "fortify" ];

            env = {
              # - libstdc++ is needed for numpy to import
              # - libpython is needed by the rust-side tests, which link against it
              LD_LIBRARY_PATH = lib.makeLibraryPath [
                pkgs.stdenv.cc.cc
                python
              ];

              POLARS_DOT_SVG_VIEWER = "${if pkgs.stdenv.hostPlatform.isLinux then "xdg-open" else "open"} %file%";
            };

            # Create the virtual environment and install the Python
            # requirements if they are missing; a no-op otherwise.
            shellHook = ''
              workspace_root=$(git rev-parse --show-toplevel)
              make -s -C "$workspace_root" .venv
              source "$workspace_root/.venv/bin/activate"
            '';
          };

          formatter = pkgs.nixfmt;
        };
    };
}
