########################################################
###    compiler flags                                ###
########################################################

SET(USE_PEDANTIC ON CACHE BOOL "Use pedantic compilation options if offered by the compiler")
SET(USE_ASAN OFF CACHE BOOL "Compile everything with AddressSanitizer enabled (GCC/Clang only)")

SET(CXX_COMPILER_ID "(unknown)" CACHE INTERNAL "The C++ compiler id")

SET(CXX_COMPILER_VERSION "0" CACHE INTERNAL "The C++ compiler version")
SET(CXX_COMPILER_VERSION_MAJOR "0" CACHE INTERNAL "The C++ compiler major version")
SET(CXX_COMPILER_VERSION_MINOR "0" CACHE INTERNAL "The C++ compiler minor version")
SET(CXX_COMPILER_VERSION_MINOR_MINOR "0" CACHE INTERNAL "The C++ compiler minor minor version")

IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

	SET(CXX_COMPILER_ID "GXX")

	## determine version number of the compiler
	EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
		OUTPUT_VARIABLE CXX_COMPILER_VERSION
	)
	STRING(STRIP ${CXX_COMPILER_VERSION} CXX_COMPILER_VERSION)

	## just to make sure that the compiler version has 3 numbers and 2 dots
	## (sometimes, g++ gives, e.g., 4.4 instead of 4.4.0)
	STRING(REGEX REPLACE "^([0-9])\\.([0-9])$" "\\1.\\2.0"
		CXX_COMPILER_VERSION ${CXX_COMPILER_VERSION})

	STRING(REGEX REPLACE "([0-9])\\.([0-9])\\.([0-9])$" "\\1"
		CXX_COMPILER_VERSION_MAJOR ${CXX_COMPILER_VERSION})
	STRING(REGEX REPLACE "([0-9])\\.([0-9])\\.([0-9])$" "\\2"
		CXX_COMPILER_VERSION_MINOR ${CXX_COMPILER_VERSION})
	STRING(REGEX REPLACE "([0-9])\\.([0-9])\\.([0-9])$" "\\3"
		CXX_COMPILER_VERSION_MINOR_MINOR ${CXX_COMPILER_VERSION})

	## -Wconversion flag for GCC
	SET(CXX_WARN_CONVERSION OFF CACHE BOOL "Enables warnings for type conversion problems (GCC only)")
	IF(CXX_WARN_CONVERSION)
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion")
	ENDIF()
	MESSAGE(STATUS "Compiler checks for conversion: ${CXX_WARN_CONVERSION}")

	IF (MT_ENABLE_CUDA)
		SET(USE_PEDANTIC OFF)
	ENDIF()

	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")

	IF(USE_PEDANTIC)
		SET(BALL_PROJECT_COMPILE_FLAGS "${BALL_PROJECT_COMPILE_FLAGS} -pedantic")
	ENDIF()

	IF(USE_ASAN)
		SET(BALL_PROJECT_COMPILE_FLAGS "${BALL_PROJECT_COMPILE_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
	ENDIF()

	## Recommended setting for eclipse, see http://www.cmake.org/Wiki/CMake:Eclipse
	IF(CMAKE_GENERATOR STREQUAL "Eclipse CDT4 - Unix Makefiles")
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmessage-length=0")
	ENDIF()

ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")

	SET(CXX_COMPILER_ID "MSVC")

	## determine version number of the compiler
	SET(CXX_COMPILER_VERSION ${MSVC_VERSION})

	STRING(REGEX REPLACE "([0-9])([0-9])([0-9])([0-9])" "\\1\\2"
		CXX_COMPILER_VERSION_MAJOR ${CXX_COMPILER_VERSION})
	STRING(REGEX REPLACE "([0-9])([0-9])([0-9])([0-9])" "\\3\\4"
		CXX_COMPILER_VERSION_MINOR ${CXX_COMPILER_VERSION})

	## disable min and max macros by default
	## see CGAL and BOOST configs for more elaborate explanations
	## NOTE: BALL_PROJECT_COMPILE_DEFNS is a CMake list (LIST(APPEND ...) is
	## used elsewhere). Append a list element, not a space-joined string —
	## otherwise ADD_DEFINITIONS receives one mangled argument and neither
	## NOMINMAX nor QT_NO_KEYWORDS actually reaches the compiler.
	LIST(APPEND BALL_PROJECT_COMPILE_DEFNS "/DNOMINMAX")

	## Silence the high-volume MSVC-only diagnostics that are noise for a
	## library exporting STL types from DLLs (BALL's design):
	##   C4251 — STL member in a dll-exported class needs dll-interface.
	##           Legitimate for every BALL container; fixing each call site
	##           would mean wrapping every std::vector<T>/std::map<...> in
	##           BALL — out of scope and not what BALL was designed for.
	##   C4244 — narrowing conversion (mostly in MSVC <utility> when STL
	##           templates are instantiated with BALL types).
	##   C4267 — size_t → smaller-type narrowing (same source as C4244).
	##   C4996 — "deprecated" — fires for the BALL_DEPRECATED class
	##           GeneticIndividual whenever the STL touches it from inside
	##           vector<>; also covers MSVC's strcpy/sprintf deprecations
	##           which are red herrings for portable code.
	## Real bugs (C4717 string.iC recursion, C4910 dllexport+extern, C4311
	## pointer truncation, C4834 nodiscard) are fixed at the source, not
	## suppressed here.
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4244 /wd4267 /wd4996")

	## if requested, produce a parallel solution
	OPTION(BALL_BUILD_SOLUTION_PARALLEL OFF)
	IF (BALL_BUILD_SOLUTION_PARALLEL)
		ADD_DEFINITIONS(/MP)
	ENDIF()

ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	
	SET(CXX_COMPILER_ID "Intel")

	## determine version number of the compiler
	EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
		OUTPUT_VARIABLE CXX_COMPILER_VERSION
	)
	STRING(STRIP ${CXX_COMPILER_VERSION} CXX_COMPILER_VERSION)
	
	## just to make sure that the compiler version has 3 numbers and 2 dots
	## (sometimes, g++ gives, e.g., 4.4 instead of 4.4.0)
	STRING(REGEX REPLACE "^([0-9][0-9])\\.([0-9])$" "\\1.\\2.0"
		CXX_COMPILER_VERSION ${CXX_COMPILER_VERSION})

	STRING(REGEX REPLACE "([0-9][0-9])\\.([0-9])\\.([0-9])$" "\\1"
		CXX_COMPILER_VERSION_MAJOR ${CXX_COMPILER_VERSION})
	STRING(REGEX REPLACE "([0-9][0-9])\\.([0-9])\\.([0-9])$" "\\2"
		CXX_COMPILER_VERSION_MINOR ${CXX_COMPILER_VERSION})
	STRING(REGEX REPLACE "([0-9][0-9])\\.([0-9])\\.([0-9])$" "\\3"
		CXX_COMPILER_VERSION_MINOR_MINOR ${CXX_COMPILER_VERSION})
	
ELSEIF(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang") # matches Clang and AppleClang
   # using Clang
	SET(CXX_COMPILER_ID "LLVM")

	## determine version number of the compiler
	EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
		OUTPUT_VARIABLE CXX_COMPILER_VERSION
	)
	STRING(STRIP ${CXX_COMPILER_VERSION} CXX_COMPILER_VERSION)

	STRING(REGEX REPLACE ".*([0-9])\\.([0-9]).*" "\\1"
		CXX_COMPILER_VERSION_MAJOR ${CXX_COMPILER_VERSION})
	STRING(REGEX REPLACE ".*([0-9])\\.([0-9]).*" "\\2"
		CXX_COMPILER_VERSION_MINOR ${CXX_COMPILER_VERSION})

	SET(CXX_COMPILER_VERSION "${CXX_COMPILER_VERSION_MAJOR}.${CXX_COMPILER_VERSION_MINOR}")

	IF(USE_ASAN)
		SET(BALL_PROJECT_COMPILE_FLAGS "${BALL_PROJECT_COMPILE_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
	ENDIF()

	# Disable "inconsistent missing override" warnings for now, as this is primarily macro-induced
	# (e.g., via BALL_EMBEDDABLE) and, unfortunately, won't be resolved until many more parts of our
	# code consistently use override specifiers.
	SET(BALL_PROJECT_COMPILE_FLAGS "${BALL_PROJECT_COMPILE_FLAGS} -Wno-inconsistent-missing-override")
ENDIF()
