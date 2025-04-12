{
  description = "A basic Nix Flake for eachDefaultSystem";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
    nixpkgs-nightly.url = "github:nixos/nixpkgs";
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

  outputs = { self, nixpkgs, nixpkgs-nightly, utils, pyproject-nix, rust-overlay }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };
        pkgs-nightly = import nixpkgs-nightly { inherit system; };

				requirements-dev  = pyproject-nix.lib.project.loadRequirementsTxt { requirements = builtins.readFile ./py-polars/requirements-dev.txt;       };
				requirements-lint = pyproject-nix.lib.project.loadRequirementsTxt { requirements = builtins.readFile ./py-polars/requirements-lint.txt;      };
				requirements-ci   = pyproject-nix.lib.project.loadRequirementsTxt { requirements = builtins.readFile ./py-polars/requirements-ci.txt;        };
				requirements-docs = pyproject-nix.lib.project.loadRequirementsTxt { requirements = builtins.readFile ./py-polars/docs/requirements-docs.txt; };

				packageOverrides = self: super: rec {
					maturin = [];
					patchelf = [];
					polars-cloud = [];
					fastexcel = pkgs-nightly.python311Packages.fastexcel;
					pytest-codspeed = pkgs-nightly.python311Packages.pytest-codspeed;
					pyiceberg = pkgs-nightly.python311Packages.pyiceberg;
					sphinx-favicon = pkgs-nightly.python311Packages.sphinx-favicon;
					sphinx-reredirects = pkgs-nightly.python311Packages.sphinx-reredirects;
					dunamai = super.dunamai.overridePythonAttrs (old: { doCheck = false; });
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
					google-auth-stubs = let
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
					in super.buildPythonPackage rec {
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
					sphinx-toolbox = let
						sphinx-jinja2-compat = super.buildPythonPackage rec {
							pname = "sphinx-jinja2-compat";
							version = "0.3.0";
							pyproject = true;

							dependencies = with super; [ whey whey-pth jinja2 markupsafe ];

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
					in super.buildPythonPackage rec {
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

						prePatch = ''
							substituteInPlace sphinx_toolbox/__init__.py      --replace 'from sphinx_toolbox.cache import cache' "# removed"
							substituteInPlace sphinx_toolbox/github/issues.py --replace 'from sphinx_toolbox.cache import cache' "# removed"
							rm -rf sphinx_toolbox/cache.py

							cat sphinx_toolbox/__init__.py
						'';

						pythonImportsCheck = [ "sphinx_toolbox" ];
					};
					typos = [];
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
						python.withPackages ( python-pkgs: let
							packages-dev  = (requirements-dev.renderers.withPackages  { inherit python; }) python-pkgs;
							packages-ci   = (requirements-ci.renderers.withPackages   { inherit python; }) python-pkgs;
							packages-docs = (requirements-docs.renderers.withPackages { inherit python; }) python-pkgs;
							packages-lint = (requirements-lint.renderers.withPackages { inherit python; }) python-pkgs;
						in (
							packages-dev ++ packages-ci ++ packages-docs ++ packages-lint
						))
					);
      in {
        devShells.default = pkgs.mkShell {
					packages = [ pythonEnv ];

				};
        packages.hello = pkgs.hello;
        packages.default = pkgs.hello;
      }
    );
}