# Building BALL / VIEW / BALLView on macOS

This document describes how to build and run the BALL library, the VIEW
visualization library, and the BALLView GUI application on macOS (Apple
Silicon) against Homebrew-provided dependencies. These are the exact,
verified commands used to build the v1.6 modernization baseline on
macOS Tahoe (arm64).

**`ball_contrib` is NOT used** by this build flow and should not be
revived — its bundled ~2016-era dependency tarballs do not build on
modern macOS toolchains. All dependencies come from Homebrew instead.

## Dependencies

Install the required libraries and build tools with Homebrew:

```sh
brew install qt@5 boost eigen fftw tbb glew open-babel lp_solve libsvm flex bison
```

`bison` and `flex` are keg-only, so their `bin` directories must be put
on `PATH` before configuring (see below).

## Configure

Create a build directory, put the keg-only `bison`/`flex` on `PATH`,
and run CMake:

```sh
mkdir -p build && cd build
export PATH="/opt/homebrew/opt/bison/bin:/opt/homebrew/opt/flex/bin:$PATH"
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/qt@5;/opt/homebrew" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBISON_EXECUTABLE=/opt/homebrew/opt/bison/bin/bison \
  -DFLEX_EXECUTABLE=/opt/homebrew/opt/flex/bin/flex \
  -DBALL_PYTHON_SUPPORT=OFF -DUSE_RTFACT=OFF -DBALL_HAS_VIEW=ON
```

## Build

From the `build` directory, build the three targets:

```sh
make BALL VIEW BALLView -j8
```

## Run

Launch BALLView from the `build` directory, pointing the data-path
environment variables at the source `data/` tree and the dynamic
linker at the freshly built libraries:

```sh
BALL_DATA_PATH=$PWD/../data BALLVIEW_DATA_PATH=$PWD/../data DYLD_LIBRARY_PATH=$PWD/lib bin/BALLView.app/Contents/MacOS/BALLView
```

## Notes

- `ball_contrib` is NOT used and should not be revived — the build
  relies entirely on Homebrew/system packages.
- Python bindings (`BALL_PYTHON_SUPPORT=OFF`), the RTfact raytracer
  (`USE_RTFACT=OFF`), OpenBabel, and TBB are currently disabled or not
  load-bearing; none of them block the build.
- `CMAKE_POLICY_VERSION_MINIMUM=3.5` is required so CMake 3.31 accepts
  the project's historical minimum policy version.
