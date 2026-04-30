# Minimal Python bindings for BALL.
#
# Wraps just enough of BALL's force-field surface to live-regenerate
# AMBER and CHARMM-EEF1 energy components from a PDB, the use case the
# proteon EVIDENT manifest's forcefield_amber_ball claim and the open
# forcefield_charmm19_internal gap point at.
#
# This subdirectory is opt-in via the top-level
#     -DBALL_BUILD_PYTHON=ON
# CMake option. Invoking BALL's normal build with the option off has
# no effect on this directory.
#
# pybind11 is fetched at configure time via FetchContent so the build
# does not require a system pybind11 install. Pinned to v2.13.6 — bump
# deliberately rather than tracking main.

cmake_minimum_required(VERSION 3.18)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(ball_py
    src/module.cpp
)

# Link to the parent project's BALL shared library target, plus its
# transitive dependencies (Boost, Qt6, Eigen, ...). BALL_DEP_LIBRARIES
# is populated by the top-level CMakeLists' BALLConfig* includes.
target_link_libraries(ball_py PRIVATE BALL ${BALL_DEP_LIBRARIES})

# BALL exports public headers under include/; the test programs in
# test/ pick them up via the parent CMakeLists' INCLUDE_DIRECTORIES.
# Repeat the same wiring locally so this subdirectory builds when
# add_subdirectory is invoked from the top-level CMakeLists.
target_include_directories(ball_py PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIR}
)

# Place the resulting .so next to the package's __init__.py-equivalent
# import path. For a `pip install`-style layout this is just the build
# directory; users who run cmake/make manually will copy the .so onto
# their PYTHONPATH.
set_target_properties(ball_py PROPERTIES
    OUTPUT_NAME "ball"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

# Sanity helper. Lets a contributor verify the wheel/import path with:
#   ctest -R python_smoke
# once the build completes. Skips silently when pytest is not on PATH.
find_package(Python COMPONENTS Interpreter QUIET)
if (Python_Interpreter_FOUND)
    add_test(
        NAME python_smoke
        COMMAND ${Python_EXECUTABLE} -m pytest
                ${CMAKE_CURRENT_SOURCE_DIR}/tests
                -v
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )
    set_tests_properties(python_smoke PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
    )
endif()
