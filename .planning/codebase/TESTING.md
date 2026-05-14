# Testing Patterns

**Analysis Date:** 2026-05-14

## Test Framework

**Custom BALL Test Framework:**
- Location: `include/BALL/CONCEPT/classTest.h`
- No external testing framework (Google Test, Catch, CppUnit) — BALL implements its own lightweight macro-based system
- Language: Pure C++ macros that generate a main() function and test harness

**Test Execution:**
- All tests are standalone executables (each `.C` file compiles to a binary)
- Location: `./test/` (295 test files)
- Compiled as individual targets via CMake: `ADD_EXECUTABLE(${test} ${test}.C)`

**Run Commands:**
```bash
# Build a specific test
cmake --build . --target Atom_test1

# Run test with no verbosity (exit code only)
./bin/TEST/Atom_test1

# Run with verbose output (-v flag)
./bin/TEST/Atom_test1 -v

# Run with maximum verbosity (-V flag)
./bin/TEST/Atom_test1 -V

# Build and run all test groups (defined in CMakeLists.txt)
cmake --build . --target BASIC_TESTS
cmake --build . --target MATHS_TESTS
cmake --build . --target SYSTEM_TESTS
```

**Assertion Library:**
- Custom macro-based assertions (not using std::assert or external libraries)
- All comparison operations implemented as preprocessor macros

## Test File Organization

**Location:**
- Co-located with source: test files in `./test/`
- Not in same directory as implementation; separate dedicated test directory
- Organized alphabetically by component name: `Atom_test1.C`, `Atom_test2.C`, `AmberFF_test.C`

**Naming:**
- Single component: `{ComponentName}_test.C` (e.g., `String_test3.C`)
- Split tests: `{ComponentName}_test{number}.C` (e.g., `Atom_test1.C`, `Atom_test2.C`, `AssignBondOrderProcessor_test1.C`, `AssignBondOrderProcessor_test2.C`)
- Corresponding to headers in `include/BALL/*/`

**Build Configuration:**
- CMake-managed: `test/CMakeLists.txt` includes test generation
- Categorized test groups: `BALL_BASIC_TESTS`, `BALL_MATHS_TESTS`, `BALL_SYSTEM_TESTS`
- Tests compiled with `-O0` (no optimization) for debugging accuracy
- Generated in: `${CMAKE_BINARY_DIR}/bin/TEST/`

## Test Structure

**Macro-Based Framework:**

All tests follow this pattern:

```cpp
// -*- Mode: C++; tab-width: 2; -*-
// vi: set ts=2:
//

#include <BALL/CONCEPT/classTest.h>
#include <BALLTestConfig.h>

///////////////////////////
// Headers for tested class
#include <BALL/KERNEL/atom.h>
#include <BALL/KERNEL/bond.h>
///////////////////////////

START_TEST(ClassName)

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

using namespace BALL;

// Test setup: variable declarations
Atom* atom = 0;

CHECK(Atom() throw())
    atom = new Atom;
    TEST_NOT_EQUAL(atom, 0)
RESULT

CHECK(void setCharge(float charge) throw())
    atom->setCharge(1.23456);
    TEST_REAL_EQUAL(atom->getCharge(), 1.23456);
RESULT

// ... more CHECK/RESULT blocks ...

END_TEST
```

**START_TEST(ClassName) Macro:**
- Location: `include/BALL/CONCEPT/classTest.h` line 128
- Generates `int main(int argc, char** argv)` function
- Sets up global TEST namespace with test state variables
- Accepts `-v` (verbose), `-V` (maximum verbose), or no args
- Wraps entire test in try/catch blocks for exception handling
- Initializes TEST::verbose, TEST::all_tests, TEST::test, TEST::version_string

**CHECK(test_name) Macro:**
- Opens a try block for one subtest
- Prints "checking {test_name}... " in verbose mode
- Manages TEST::test (current subtest status) and TEST::newline flag
- All assertions between CHECK and RESULT contribute to this subtest's result

**RESULT Macro:**
- Closes the try block opened by CHECK
- Catches all BALL exceptions (GeneralException, FileNotFound) and std::exception
- Catches all other exceptions with generic handler
- Prints "passed" or "FAILED" based on TEST::test state
- Updates TEST::all_tests for final exit code

**END_TEST Macro:**
- Closes the global try block (opened by START_TEST)
- Catches global exceptions (uncaught exceptions fail entire test)
- Deletes temporary files from TEST::tmp_file_list
- Prints "PASSED" or "FAILED" at exit
- Returns 0 on success, 1 on failure

## Test Macros and Assertions

**Basic Equality:**
```cpp
TEST_EQUAL(a, b)         // a == b; prints line, values if fails
TEST_NOT_EQUAL(a, b)     // !(a == b); prints forbidden value if fails
```

**Floating Point:**
```cpp
TEST_REAL_EQUAL(a, b)    // fabs(a - b) < TEST::precision (default 1e-6)
                         // Uses TEST::precision for tolerance
```

**Exception Testing:**
```cpp
TEST_EXCEPTION(ExceptionType, command)
    // Executes command, verifies ExceptionType is thrown
    // Sets TEST::exception = 1 if correct, 2 for wrong exception, 3 for other
    // TEST::exception = 0 if no exception thrown

#ifdef BALL_DEBUG
TEST_PRECONDITION_EXCEPTION(command)
    // Tests for Exception::Precondition (debug mode only)
    // No-op if not in DEBUG mode
#endif
```

**Status Messages:**
```cpp
STATUS("message text")   // Prints "status: message text" in verbose mode (-V)
                         // Used for progress output during long operations
```

**Temporary Files:**
```cpp
NEW_TMP_FILE(filename)              // Creates temp file, stores name for cleanup
NEW_TMP_FILE_WITH_SUFFIX(filename, suffix)  // Creates with specific suffix
    // All temp files auto-deleted after tests complete
```

**Sleep:**
```cpp
SLEEP_FOR_MSECS(ms)      // Block execution for milliseconds
                         // Uses std::this_thread::sleep_for
```

**Precision Control:**
```cpp
TEST::precision = 1e-6;  // Set floating point tolerance (default)
```

## Actual Test Examples

**Simple Method Test (`Atom_test1.C`):**
```cpp
CHECK(Atom() throw())
    atom = new Atom;
    TEST_NOT_EQUAL(atom, 0)
RESULT

CHECK(void setCharge(float charge) throw())
    atom->setCharge(1.23456);
    TEST_REAL_EQUAL(atom->getCharge(), 1.23456);
RESULT

CHECK(float getCharge() const throw())
    Atom a;
    TEST_REAL_EQUAL(a.getCharge(), 0.0);
RESULT
```

**Complex State Test (`Atom_test2.C`):**
```cpp
Atom* atom = new Atom;
Atom* atom3 = new Atom(*atom);
Atom* atom4 = new Atom(PTE[Element::LITHIUM], "TESTNAME2", "TESTTYPE2", 
                       23, test_vector, test_vector, null_vector, 3.23456, 4.34567);

CHECK(Bond* createBond(Atom& atom) throw())
    atom->createBond(*atom3);
    atom3->getBond(*atom);
    TEST_EQUAL(atom->countBonds(), 1)
    TEST_EQUAL(atom3->countBonds(), 1)
    TEST_EQUAL(atom->getBond(*atom3), atom3->getBond(*atom))
    atom->createBond(*atom4);
    atom->createBond(*atom4);
    bond = atom->createBond(*atom);
    TEST_EQUAL(bond, 0);
RESULT
```

**String Conversion Test (`String_test3.C`):**
```cpp
CHECK(const String& operator = (int i) throw())
    s4 = (int)-19;
    TEST_EQUAL(s4, "-19")
RESULT

CHECK(const String& operator = (float f) throw())
    s4 = (float)-123.456;
    TEST_REAL_EQUAL(atof(s4.c_str()), -123.456)
RESULT
```

## Test Data and Fixtures

**Global Test Objects:**
```cpp
// Declared at START_TEST level (before first CHECK)
Atom* atom = 0;
Vector3 test_vector(1, 2, 3);
Vector3 null_vector(0, 0, 0);
Molecule molecule;
Fragment fragment;
```

**Object Setup:**
- Test objects created and modified within CHECK blocks
- State persists across multiple subtests (all_tests variable tracks overall result)
- Cleanup happens implicitly or via destructor calls

**No Test Fixtures:**
- BALL's framework does not provide setup/teardown fixtures
- Each test is independent (though objects may be reused across CHECK blocks)
- Temporary files managed via NEW_TMP_FILE macro

## Mocking

**No Mocking Framework:**
- No mocks, stubs, or test doubles (Google Mock not used)
- Tests work with real objects and real dependencies
- External dependencies (files, third-party libraries) called directly

**File Testing:**
```cpp
String filename;
NEW_TMP_FILE(filename)   // Create real temporary file
// Read/write to real filesystem
// File automatically deleted after test
```

**What NOT to Mock:**
- Any BALL class hierarchy (test with real objects)
- Standard library containers (std::vector, std::string, etc.)
- Relationships between classes (bonds between atoms tested directly)

**What IS "Mocked":**
- File I/O: NEW_TMP_FILE provides real temp files instead of mock I/O
- External processes: None (BALL is a library)

## Test Organization by Type

**Unit Tests:**
- Scope: Individual class or method
- Approach: Test constructor, getters, setters, operations
- Example: `Atom_test1.C` tests Atom construction and charge accessors
- Location: `test/{ClassName}_test*.C`

**Integration Tests:**
- Scope: Multiple classes working together
- Approach: Create relationships and test interaction
- Example: `Atom_test2.C` tests Bond creation and Atom container relationships
- Location: Same test files, different CHECK blocks

**No E2E Tests:**
- End-to-end testing not structured in this framework
- High-level integration possible through multiple components in same test

## Coverage

**Coverage Enforcement:**
- No explicit code coverage requirements detected (BALL_TEST_VERBOSE option only)
- No coverage reporting configuration in CMakeLists.txt
- Tests run to completion, but no coverage metrics generated

**View Coverage:**
- Not configured in visible build system
- Would require external tool (gcov, llvm-cov) if desired

## Verbose Output Control

**Verbosity Levels:**

```
No argument:  Only final PASSED/FAILED printed, exit code only
-v flag:      Subtest names and failures printed
-V flag:      All subtests, all STATUS messages, all details printed
```

**Example Output (no args):**
```
PASSED
```

**Example Output (-v):**
```
checking Atom() throw()... passed
checking void setCharge(float charge) throw()... passed
checking float getCharge() const throw()... passed
```

**Example Output (-V):**
```
checking Atom() throw()... passed
checking void setCharge(float charge) throw()... passed
status (line 42): Processing element data
  (line 42 TEST_REAL_EQUAL(a.getCharge(), 0.0): got 0, expected 0) +
  ... detailed messages ...
PASSED
```

## Compilation Flags

**Debug vs Release:**
- Tests compiled with `-O0` (no optimization) regardless of build type
- CMAKE_CXX_FLAGS_RELEASE saved and restored
- Allows precise debugging and correct assertion behavior

**Include Paths:**
- `${PROJECT_BINARY_DIR}` for generated config (BALLTestConfig.h)
- BALL library headers via include directories
- Test support libraries: pthread on UNIX

## Special Features

**Exception Handling in Framework:**
- BALL exceptions caught with file/line information preserved
- std::exception caught and message printed
- Generic catch-all for unexpected exception types
- Failed exceptions mark TEST::all_tests = false, continue to next test

**Temporary Files:**
- Auto-cleanup: all files in TEST::tmp_file_list deleted before exit
- Only deleted if TEST::verbose < 1 (only in non-verbose mode)
- Useful for file I/O testing without manual cleanup

**State Inspection:**
```cpp
TEST::verbose       // 0 (none), 1 (-v), 2 (-V)
TEST::all_tests     // Overall test result (bool)
TEST::test          // Current subtest result (bool)
TEST::this_test     // Last assertion result (bool)
TEST::exception     // Exception handling state
```

---

*Testing analysis: 2026-05-14*
