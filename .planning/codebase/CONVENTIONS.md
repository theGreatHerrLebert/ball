# Coding Conventions

**Analysis Date:** 2026-05-14

## Naming Patterns

**Files:**
- Headers: `.h` extension with uppercase namespace/module directories (e.g., `BALL/KERNEL/atom.h`)
- Implementation: `.C` extension (C++ convention), matching header names (e.g., `source/QSAR/Model.C`)
- Test files: `{ComponentName}_test.C` or `{ComponentName}_test{number}.C` (e.g., `Atom_test1.C`, `AssignBondOrderProcessor_test1.C`)

**Classes:**
- PascalCase: `Atom`, `Bond`, `Model`, `LibsvmModel`, `SVRModel`
- BALL_EXPORT macro used for shared library visibility (e.g., `class BALL_EXPORT Atom`)
- Nested namespaces use PascalCase for modules: `BALL::QSAR::Model`

**Functions/Methods:**
- camelCase: `setCharge()`, `getCharge()`, `createBond()`, `destroyBond()`, `optimizeParameters()`
- Getter methods: `get*` prefix: `getElement()`, `getFragment()`, `getType()`
- Setter methods: `set*` prefix: `setCharge()`, `setElement()`, `setName()`
- Predicate methods: `has*`, `is*`: `hasBond()`, `countBonds()`
- Action methods: `create*`, `destroy*`: `createBond()`, `destroyBond()`

**Variables:**
- Local and parameter variables: camelCase: `atom`, `bond`, `descriptor_matrix`, `substance_names`
- Member variables: camelCase with trailing underscore: `descriptor_matrix_`, `y_transformations_`, `model_val`, `descriptor_IDs_`, `type_`, `substance_names_`
- Global/static: camelCase or UPPER_CASE for constants: `BALL_ATOM_DEFAULT_CHARGE`

**Types/Enums:**
- PascalCase: `FullNameType`, `Property`
- Enum values: UPPER_CASE: `UNKNOWN_TYPE`, `MAX_NUMBER_OF_BONDS`, `ANY_TYPE`
- Typedef'd types use PascalCase: `Type`, `Index`

**Constants/Macros:**
- UPPER_CASE with underscores: `BALL_ATOM_DEFAULT_ELEMENT`, `BALL_CREATE_DEEP`, `MAX_NUMBER_OF_BONDS`
- Macros defined in headers for default values use `BALL_{CLASS}_DEFAULT_*` pattern

## Code Style

**Formatting:**
- Tabs: 2-space indentation (controlled via `// -*- Mode: C++; tab-width: 2; -*-`)
- Vi settings: `// vi: set ts=2:` present in all source files
- Brace style: Allman (opening brace on new line for blocks, same line for function parameters)
- Line continuation: Indented consistently under function parameters

**Documentation:**
- All headers start with: `// -*- Mode: C++; tab-width: 2; -*-` and `// vi: set ts=2:`
- Classes documented with multi-line /// comments preceding class definition
- Doxygen-style documentation: `/** ... */` blocks with `@param`, `@return`, `@see` tags
- Method documentation includes parameter descriptions and return value documentation

**Include Guards:**
- Pattern: `#ifndef BALL_{MODULE}_{FILE}_H` and `#define BALL_{MODULE}_{FILE}_H`
- Example: `#ifndef BALL_KERNEL_ATOM_H` in `include/BALL/KERNEL/atom.h`
- Closing: `#endif // BALL_{MODULE}_{FILE}_H`

**Conditional Includes:**
- Check against define before including: `#ifndef BALL_CONCEPT_COMPOSITE_H` then `#	include <BALL/CONCEPT/composite.h>`
- Uses tab-indented include (`#	`) for included guards

## Import Organization

**Header Includes Order:**
1. Local include guards (ifndef/define)
2. Internal BALL headers (with guard checks): `#ifndef BALL_*_H` `#	include <BALL/.../>`
3. Standard library includes: `#include <vector>`, `#include <set>`, `#include <string>`
4. Third-party includes: `#include <Eigen/Core>`

**Using Declarations:**
- `using namespace BALL;` declared in most source files after includes
- `using std::string;` for specific STL types in some files
- Nested namespaces: Files in `QSAR` module use `namespace BALL { namespace QSAR { ... } }`

## Error Handling

**Exceptions:**
- Throw BALL exception types: `Exception::InconsistentUsage(__FILE__, __LINE__, "message")`
- Other BALL exceptions: `Exception::GeneralException`, `Exception::FileNotFound`, `Exception::IndexOverflow`
- Exceptions carry file, line, and message information

**Validation:**
- Precondition checks before operations: check sizes, verify non-null pointers
- Example: `if (descriptor_matrix_.cols() == 0) { throw Exception::InconsistentUsage(...); }`
- Return values checked before use (e.g., pointer validity after allocation)

**Return Values:**
- Methods may return `nullptr` or `0` to indicate failure: `Bond* createBond(Atom& atom)` returns `0` if bond cannot be created
- Boolean flags used for operation success: operations return `bool` for success/failure

## Comments

**When to Comment:**
- Complex algorithms or non-obvious logic get inline comments
- Variable purpose explanations when name is insufficient: `int t = 0; // index in line of training data`
- Disabled code blocks commented with reason: `// free(prob); // handled by svm_free_and_destroy_model`

**Documentation Style:**
- Doxygen documentation mandatory for public APIs
- Brief description on first line: `/** Default constructor.`
- Parameter documentation: `@param   name description`
- Return documentation: `@return  Type - description`
- Cross-reference documentation: `@see ClassName::methodName`

**File Headers:**
- All source and header files start with modeline comments
- No author/date information in individual files (likely in project metadata)

## Function Design

**Size:**
- Methods typically 10-40 lines, some models methods 50+ lines for complex algorithms
- Preference for smaller, focused methods with clear responsibility

**Parameters:**
- Const references for input objects: `const Model& m`, `const QSARData& q`
- Output parameters by reference: `void readMatrix(Eigen::MatrixXd& mat, ...)`
- Boolean parameters for feature flags: `bool deep = true`, `bool transform`
- Type casts in parameters: `(double*)malloc(...)` for C-style APIs

**Return Values:**
- Methods return by value for small types: `float getCharge() const`
- Methods return by const reference for large objects: `const Eigen::MatrixXd* getDescriptorMatrix()`
- Void returns for operations: `void setCharge(float charge)`
- Pointers returned for object creation: `Bond* createBond(Atom& atom)`

## Member Variables

**Encapsulation:**
- Private members accessed via getter/setter pairs
- Protected members for inherited classes
- Member variables with trailing underscore (private/protected): `descriptor_matrix_`, `Y_`, `type_`
- Public attributes for simple data containers: `const QSARData* data;`

**Initialization:**
- Member initialization in constructor bodies (not modern initialization lists)
- Default values set in class definitions for constants
- Example: `type_="SVR"; svm_train_result_ = NULL;`

## Namespace Organization

**BALL Core:**
- Everything in `namespace BALL { ... }`
- Submodules: `namespace BALL { namespace QSAR { ... } }`
- Modules organized by functionality: `KERNEL`, `CONCEPT`, `QSAR`, `MATHS`, `SYSTEM`, `DATATYPE`

**Using Declarations:**
- Global `using namespace BALL;` in source files
- Specific `using` for STL: `using std::string;` or `using namespace std;`
- Scoped usage in headers minimized

---

*Convention analysis: 2026-05-14*
