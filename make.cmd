@echo off

set base_path=%~dp0
set base_path=%base_path:~0,-1%
set py_path=%base_path%\py-polars
set VENV=%base_path%\.venv
set VENV_BIN=%VENV%\Scripts

:: Ensure py-polars/rust-toolchain.toml is symlinked or hardlinked
:: This commonly fails on Windows, since users must both have administrator
:: access and also must enable symbolic links when installing Git for Windows
set SYMLINK=
for /F "tokens=* USEBACKQ" %%f in (
    `fsutil reparsepoint query "%py_path%\rust-toolchain.toml" ^| find "Symbolic Link" 2^>^&1 ^>nul ^&^& echo 1`
) do set SYMLINK=%%f

if not '%SYMLINK%'=='1' (
    set HARDLINK=
    for /F "tokens=* USEBACKQ" %%f in (
        `fsutil hardlink list "%base_path%\rust-toolchain.toml" ^| find "polars\py-polars\rust-toolchain.toml" 2^>^&1 ^>nul ^&^& echo 1`
    ) do set HARDLINK=%%f

    if not '%HARDLINK%'=='1' (
        if not exist "%base_path%\py-polars\rust-toolchain" (
            echo py-polars\rust-toolchain.toml did not successfully link to rust-toolchain.toml; creating `rust-toolchain` override file.
            mklink /H "%py_path%\rust-toolchain" "%base_path%\rust-toolchain.toml"
        )
    )
)

if %1==.venv                   call :.venv
if %1==requirements            call :requirements
if %1==build                   call :build
if %1==build-debug-opt         call :build-debug-opt
if %1==build-debug-opt-subset  call :build-debug-opt-subset
if %1==build-opt               call :build-opt
if %1==build-release           call :build-release
if %1==build-native            call :build-native
if %1==build-debug-opt-native  call :build-debug-opt-native
if %1==clippy                  call :clippy
if %1==clippy-default          call :clippy-default
if %1==fmt                     call :fmt
if %1==pre-commit              call :pre-commit
if %1==clean                   call :clean
if %1==help                    call :help
exit /b

:.venv
:: Set up Python virtual environment and install requirements
python -m venv %VENV%
call :requirements
exit /b

:requirements
:: Install/refresh Python project requirements
%VENV_BIN%\python -m pip install --upgrade pip
%VENV_BIN%\python -m pip install --upgrade -r %py_path%\requirements-dev.txt
%VENV_BIN%\python -m pip install --upgrade -r %py_path%\requirements-lint.txt
%VENV_BIN%\python -m pip install --upgrade -r %py_path%\docs\requirements-docs.txt
%VENV_BIN%\python -m pip install --upgrade -r %base_path%\docs\requirements.txt
::powershell -command "iwr https://dprint.dev/install.ps1 -useb | iex"
exit /b

:build
:: Compile and install Python Polars for development
call :.venv
set CONDA_PREFIX=
%VENV_BIN%\activate && maturin develop -m %py_path%\Cargo.toml
exit /b

:build-debug-opt
:: Compile and install Python Polars with minimal optimizations turned on
call :.venv
set CONDA_PREFIX=
%VENV_BIN%\activate && maturin develop -m %py_path%\Cargo.toml --profile opt-dev
exit /b

:build-debug-opt-subset
:: Compile and install Python Polars with minimal optimizations turned on and no default features
call :.venv
set CONDA_PREFIX=
%VENV_BIN%\activate && maturin develop -m %py_path%\Cargo.toml --no-default-features --profile opt-dev
exit /b

:build-opt
::Compile and install Python Polars with nearly full optimization on and debug assertions turned off, but with debug symbols on
call :.venv
set CONDA_PREFIX=
%VENV_BIN%\activate && maturin develop -m %py_path%\Cargo.toml --profile debug-release
exit /b

:build-release
:: Compile and install a faster Python Polars binary with full optimizations
call :.venv
set CONDA_PREFIX=
%VENV_BIN%\activate && maturin develop -m %py_path%\Cargo.toml --release
exit /b

:build-native
:: Same as build, except with native CPU optimizations turned on
call :.venv
set CONDA_PREFIX=
%VENV_BIN%\activate && maturin develop -m %py_path%\Cargo.toml -- -C target-cpu=native
exit /b

:build-debug-opt-native
:: Same as build-debug-opt, except with native CPU optimizations turned on
call :.venv
set CONDA_PREFIX=
%VENV_BIN%\activate && maturin develop -m %py_path%\Cargo.toml --profile opt-dev -- -C target-cpu=native
exit /b

:build-opt-native
:: Same as build-opt, except with native CPU optimizations turned on
call :.venv
set CONDA_PREFIX=
%VENV_BIN%\activate && maturin develop -m %py_path%\Cargo.toml --profile debug-release -- -C target-cpu=native
exit /b

:build-release-native
:: Same as build-release, except with native CPU optimizations turned on
call :.venv
set CONDA_PREFIX=
%VENV_BIN%\activate && maturin develop -m %py_path%\Cargo.toml --release -- -C target-cpu=native
exit /b

:clippy
:: ## Run clippy with all features
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
exit /b

:clippy-default
::Run clippy with default features
cargo clippy --all-targets --locked -- -D warnings
exit /b

:fmt
:: Run autoformatting and linting
%VENV_BIN%\ruff check %base_path%  &^
%VENV_BIN%\ruff format %base_path% &^
cargo fmt --all &^
::if exist %USERPROFILE%\.dprint\bin\dprint.exe (
::    %USERPROFILE%\.dprint\bin\dprint.exe fmt --excludes py-polars/rust-toolchain.toml
::) &^
%VENV_BIN%\typos &^
exit /b

:pre-commit
:: Run all code quality checks
call :fmt
call :clippy
call :clippy-default
exit /b

:clean
:: Clean up caches and build artifacts
if exist %base_path%\.venv ( rmdir %base_path%\.venv /s /q ) &^
if exist %base_path%\target ( rmdir %base_path%\target /s /q ) &^
if exist %base_path%\Cargo.lock ( del %base_path%\Cargo.lock /s /q ) &^
cargo clean
call %py_path%\make clean
exit /b

:help
:: Display this help screen
echo Available commands:
echo  .venv                  Set up Python virtual environment and install requirements
echo  build                  Compile and install Python Polars for development
echo  build-debug-opt        Compile and install Python Polars with minimal optimizations turned on
echo  build-debug-opt-native Same as build-debug-opt, except with native CPU optimizations turned on
echo  build-debug-opt-subset Compile and install Python Polars with minimal optimizations turned on and no default features
echo  build-native           Same as build, except with native CPU optimizations turned on
echo  build-opt              Compile and install Python Polars with nearly full optimization on and debug assertions turned off, but with debug symbols on
echo  build-opt-native       Same as build-opt, except with native CPU optimizations turned on
echo  build-release          Compile and install a faster Python Polars binary with full optimizations
echo  build-release-native   Same as build-release, except with native CPU optimizations turned on
echo  clean                  Clean up caches and build artifacts
echo  clippy                 Run clippy with all features
echo  clippy-default         Run clippy with default features
echo  fmt                    Run autoformatting and linting
echo  help                   Display this help screen
echo  pre-commit             Run all code quality checks
echo  requirements           Install/refresh Python project requirements
