@echo off

set py_path=%~dp0
set py_path=%py_path:~0,-1%
for %%i in ("%~dp0..") do set base_path=%%~fi
set VENV=%base_path%\.venv
set VENV_BIN=%VENV%\Scripts

if %1==.venv                   call :.venv %1
if %1==requirements            call :requirements %1
if %1==build                   call :build %1
if %1==build-debug-opt         call :build-debug-opt %1
if %1==build-debug-opt-subset  call :build-debug-opt-subset %1
if %1==build-opt               call :build-opt %1
if %1==build-release           call :build-release %1
if %1==build-native            call :build-native %1
if %1==build-debug-opt-native  call :build-debug-opt-native %1
if %1==clippy                  call :clippy %1
if %1==clippy-default          call :clippy-default %1
if %1==fmt                     call :fmt %1
if %1==pre-commit              call :pre-commit %1
if %1==test                    call :test %1
if %1==doctest                 call :doctest %1
if %1==test-all                call :test-all %1
if %1==coverage                call :coverage %1
if %1==clean                   call :clean %1
if %1==help                    call :help %1
exit /b

:.venv
:: Set up virtual environment and install requirements
call %base_path%\make %1
exit /b

:requirements
:: Install/refresh all project requirements
::call %base_path%\make %1
exit /b

:build
:: Compile and install Polars for development
call %base_path%\make %1
exit /b

:build-debug-opt
:: Compile and install Polars with minimal optimizations turned on
call %base_path%\make %1
exit /b

:build-debug-opt-subset
:: Compile and install Polars with minimal optimizations turned on and no default features
call %base_path%\make %1
exit /b

:build-opt
:: Compile and install Polars with nearly full optimization on and debug assertions turned off, but with debug symbols on
call %base_path%\make %1
exit /b

:build-release
:: Compile and install a faster Polars binary with full optimizations
call %base_path%\make %1
exit /b

:build-native
:: Same as build, except with native CPU optimizations turned on
call %base_path%\make %1
exit /b

:build-debug-opt-native
:: Same as build-debug-opt, except with native CPU optimizations turned on
call %base_path%\make %1
exit /b

:build-opt-native
:: Same as build-opt, except with native CPU optimizations turned on
call %base_path%\make %1
exit /b

:build-release-native
:: Same as build-release, except with native CPU optimizations turned on
call %base_path%\make %1
exit /b

:fmt
:: Run autoformatting and linting
call %base_path%\make %1
exit /b

:clippy
:: Run clippy
call %base_path%\make %1
exit /b

:pre-commit
:: Run all code quality checks
call %base_path%\make %1
exit /b

:test
:: Run fast unittests
%VENV_BIN%\pytest %pypath%/tests -n auto --dist loadgroup
exit /b

:doctest
:: Run doctests
%VENV_BIN%\python %pypath%/tests/docs/run_doctest.py
%VENV_BIN%\pytest %pypath%/tests/docs/test_user_guide.py -m docs
exit /b

:test-all
:: Run all tests
%VENV_BIN%\pytest %pypath%/tests -n auto --dist loadgroup -m "slow or not slow"
%VENV_BIN%\python %pypath%/tests/docs/run_doctest.py
exit /b

:coverage
:: Run tests and report coverage
%VENV_BIN%\pytest %pypath%/tests --cov -n auto --dist loadgroup -m "not benchmark"
exit /b

:clean
:: Clean up caches and build artifacts
if exist %py_path%\target ( rmdir %py_path%\target /s /q )
if exist %py_path%\target ( rmdir %py_path%\target /s /q )
if exist %py_path%\docs\build ( rmdir %py_path%\docs\build /s /q )
if exist %py_path%\docs\source\reference\api ( rmdir %py_path%\docs\source\reference\api /s /q )
if exist %py_path%\.hypothesis ( rmdir %py_path%\.hypothesis /s /q )
if exist %py_path%\.mypy_cache ( rmdir %py_path%\.mypy_cache /s /q )
if exist %py_path%\.pytest_cache ( rmdir %py_path%\.pytest_cache /s /q )
if exist %py_path%\.ruff_cache ( rmdir %py_path%\.ruff_cache /s /q )
if exist %py_path%\.coverage ( rmdir %py_path%\.coverage /s /q )
if exist %py_path%/coverage.xml ( del %py_path%/coverage.xml /s /q )
if exist %py_path%/polars/polars.abi3.so ( del %py_path%/polars/polars.abi3.so /s /q )
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul
cargo clean
exit /b

:help
:: Display this help screen
echo Available Commands:
echo  .venv                  Set up virtual environment and install requirements
echo  build                  Compile and install Polars for development
echo  build-debug-opt        Compile and install Polars with minimal optimizations turned on
echo  build-debug-opt-native Same as build-debug-opt, except with native CPU optimizations turned on
echo  build-debug-opt-subset Compile and install Polars with minimal optimizations turned on and no default features
echo  build-native           Same as build, except with native CPU optimizations turned on
echo  build-opt              Compile and install Polars with nearly full optimization on and debug assertions turned off, but with debug symbols on
echo  build-opt-native       Same as build-opt, except with native CPU optimizations turned on
echo  build-release          Compile and install a faster Polars binary with full optimizations
echo  build-release-native   Same as build-release, except with native CPU optimizations turned on
echo  clean                  Clean up caches and build artifacts
echo  clippy                 Run clippy
echo  coverage               Run tests and report coverage
echo  doctest                Run doctests
echo  fmt                    Run autoformatting and linting
echo  help                   Display this help screen
echo  pre-commit             Run all code quality checks
echo  requirements           Install/refresh all project requirements
echo  test                   Run fast unittests
echo  test-all               Run all tests
