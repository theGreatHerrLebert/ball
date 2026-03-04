########################################################
#
# CMake configuration for BALL
# http://www.ball-project.org
#
########################################################

CMAKE_MINIMUM_REQUIRED(VERSION 3.30 FATAL_ERROR)

PROJECT("BALL" VERSION 1.5.0)


########################################################
# General build system settings
########################################################

# Use folders to group TARGTES using the folder target property in IDEs
SET(USE_FOLDERS TRUE)

# Automatically search lib64 folders
SET(FIND_LIBRARY_USE_LIB64_PATHS TRUE)

# Set default build type to Release
IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Release)
ENDIF()

# Add cmake directory from BALL source tree to CMAKE_MODULE_PATH
SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PROJECT_SOURCE_DIR}/cmake")

# Add include directoy from BALL source tree
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include")

# Add include directoy from build tree for configured files
INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}/include")

# Include BALL specific CMake macros
INCLUDE(cmake/BALLMacros.cmake)

# BALL specific compiler and linker settings
SET(BALL_PROJECT_COMPILE_FLAGS "")
SET(BALL_PROJECT_COMPILE_DEFNS "")
INCLUDE(BALLCompilerSpecific)

# Determine BALL configuration
INCLUDE(BALLConfiguration)

INCLUDE(GNUInstallDirs)

# We do not have a Component-based installer in MacOS
IF(NOT APPLE)
	INCLUDE(BALLComponents)
ENDIF()

# License choice: LGPL or GPL build
SET(BALL_LICENSE "LGPL" CACHE STRING "BALL licensing model: LGPL (default) or GPL")

IF(${BALL_LICENSE} STREQUAL "LGPL")
	SET(BALL_LICENSE_LGPL TRUE)
	SET(BALL_LICENSE_GPL  FALSE)
ELSEIF(${BALL_LICENSE} STREQUAL "GPL")
	SET(BALL_LICENSE_LGPL FALSE)
	SET(BALL_LICENSE_GPL  TRUE)
ELSE()
	MESSAGE(FATAL_ERROR "BALL_LICENSE can either be 'LGPL' (default) or 'GPL'")
ENDIF()



########################################################
# Search and configure third party dependencies
########################################################

## Extra libraries we want to link into BALL
SET(BALL_DEP_LIBRARIES)

## Flex and Bison
## This ugly hack is required as long as we cannot tell cmake to set an environment variable for bison...
FIND_PACKAGE(BISON)
IF (NOT BISON_FOUND)
	MESSAGE(FATAL_ERROR "Could not find bison parser generator!")
ENDIF()

FIND_PACKAGE(FLEX)
IF (NOT FLEX_FOUND)
	MESSAGE(FATAL_ERROR "Could not find flex scanner generator!")
ENDIF()
## Figure out if the used flex version supports ...lex_destroy()
MESSAGE(STATUS "Checking for yylex_destroy support in flex")
STRING(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+).*$" "\\1;\\2;\\3"
	FLEX_VERSION_NUMERIC ${FLEX_VERSION})
LIST(GET FLEX_VERSION_NUMERIC 2 FLEX_PATCH_LEVEL)
IF (FLEX_PATCH_LEVEL GREATER 8)
	MESSAGE(STATUS "Checking for yylex_destroy support in flex - found")
	SET(BALL_HAS_YYLEX_DESTROY TRUE)
ELSE()
	MESSAGE(STATUS "Checking for yylex_destroy support in flex - Not found")
	SET(BALL_HAS_YYLEX_DESTROY FALSE)
ENDIF()


## Boost
INCLUDE(BALLConfigBoost)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
LIST(APPEND BALL_DEP_LIBRARIES ${Boost_LIBRARIES})


IF(${BALL_LICENSE_GPL})

	# FFTW
	OPTION(USE_FFTWD "Use double precision FFTW if found" ON)
	OPTION(USE_FFTWF "Use single precision FFTW if found" ON)
	OPTION(USE_FFTWF "Use long double precision FFTW if found" ON)
	OPTION(USE_FFTW_THREADS "Try to find FFTW with thread support" OFF)

	FIND_PACKAGE(FFTW)
	IF(FFTW_FOUND)
		INCLUDE_DIRECTORIES(${FFTW_INCLUDE_DIR})
		INCLUDE(BALLConfigFFTW)
	ENDIF()

	#Set the default complex precision to float if the setting for this value
	#could not be determined earlier
	IF(NOT BALL_COMPLEX_PRECISION)
		SET(BALL_COMPLEX_PRECISION "float")
	ENDIF()
ENDIF()


## lpsolve
OPTION(USE_LPSOLVE "Compile with lpsolve support, if lpsolve can be found" ON)

FIND_PACKAGE(LPSolve)
IF (LPSOLVE_FOUND)
	SET(BALL_HAS_LPSOLVE TRUE)
	INCLUDE_DIRECTORIES(${LPSOLVE_INCLUDE_DIR})
ENDIF()

## libsvm
OPTION(USE_LIBSVM "Compile with libsvm support if libsvm can be found" ON)

IF(USE_LIBSVM)
	FIND_PACKAGE(libSVM)
	IF (LIBSVM_FOUND)
		SET(BALL_HAS_LIBSVM TRUE)
		INCLUDE_DIRECTORIES(${LIBSVM_INCLUDE_DIRS})
	ENDIF()
ENDIF()

## QT
# Enable automatic invocation of moc if required
SET(CMAKE_AUTOMOC ON)
SET(CMAKE_AUTOMOC_MOC_OPTIONS "-DBALL_DEPRECATED=") # Fix issues on some machines where moc would ignore classes with
                                                    # multiple attribute specifications via global macros

SET(QTDIR "" CACHE PATH "Path to the qt installation (optional)")
SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${QTDIR})

IF(QTDIR)
	INCLUDE_DIRECTORIES(BEFORE "${QTDIR}/include")
ENDIF()

SET(QT_MIN_VERSION 6.2)

FIND_PACKAGE(Qt6 ${QT_MIN_VERSION} REQUIRED COMPONENTS Core Core5Compat Network Xml)

LIST(APPEND BALL_DEP_LIBRARIES Qt6::Core
		                       Qt6::Core5Compat
                               Qt6::Network
                               Qt6::Xml)

SET(EIGEN3_VERSION_OK "3.0.0")
FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

IF(APPLE)
	FIND_PROGRAM(MACDEPLOYQT_EXECUTABLE macdeployqt PATHS ${BALL_CONTRIB_PATH}/bin)

	IF(${MACDEPLOYQT_EXECUTABLE} STREQUAL "MACDEPLOYQT_EXECUTABLE-NOTFOUND")
		MESSAGE(WARNING "MacOS Bundle generation not possible: macdeployqt not found")
	ELSE()
		SET(QT_DEPLOY_EXECUTABLE ${MACDEPLOYQT_EXECUTABLE})
	ENDIF()
ENDIF()


## replace any variables in config.h.in with current values
SET(CONFIGURED_CONFIG_H ${PROJECT_BINARY_DIR}/include/BALL/CONFIG/config.h)
CONFIGURE_FILE(cmake/config.h.in ${CONFIGURED_CONFIG_H})

## build a config.h file for our tests to not disturb the rest of BALL
CONFIGURE_FILE(cmake/BALLTestConfig.h.in ${PROJECT_BINARY_DIR}/test/BALLTestConfig.h)

## build a config.h file for our benchmarks to not disturb the rest of BALL
CONFIGURE_FILE(cmake/BALLBenchmarkConfig.h.in ${PROJECT_BINARY_DIR}/source/BENCHMARKS/BALLBenchmarkConfig.h)


SET(BALL_EXECUTABLES "")
########################################################
###    BUILD the lib																 ###
########################################################


# Add BALL project specific compiler flags and definitions
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BALL_PROJECT_COMPILE_FLAGS}")
ADD_DEFINITIONS("${BALL_PROJECT_COMPILE_DEFNS}")


## are we building a shared or static lib?! (BOTH within
## the same BUILD-tree is NOT possible currently!!)
SET(BUILD_SHARED_LIBS TRUE)

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

## big include file for headers and C files, which fills the BALL_sources variable
INCLUDE(BALLIncludes)

## mark header files as headers...
SET_PROPERTY(SOURCE ${BALL_headers} APPEND PROPERTY HEADER_FILE_ONLY TRUE)

## add library target
ADD_LIBRARY(BALL SHARED ${BALL_sources} ${BALL_headers})

## all the dependency libraries are linked into libBALL.so, except Qt which are still dynamic
LIST(APPEND BALL_DEP_LIBRARIES
	${BOOST_LIBRARIES}
	${FFTW_LIBRARIES}
	${LPSOLVE_LIBRARIES}
)


IF(UNIX)
	LIST(APPEND BALL_DEP_LIBRARIES "pthread")
ENDIF()

IF(LIBSVM_FOUND)
	LIST(APPEND BALL_DEP_LIBRARIES ${LIBSVM_LIBRARIES})
ENDIF()

SET(BALL_COMPILE_FLAGS)
SET(BALL_LINK_FLAGS)

SET_TARGET_PROPERTIES(BALL PROPERTIES
	SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}"
)

IF(CXX_COMPILER_ID STREQUAL "GXX")
	IF(ENABLE_COVERAGE_TESTING)
		SET(BALL_COMPILE_FLAGS "${BALL_COMPILE_FLAGS} -fprofile-arcs -ftest-coverage")
		SET(BALL_LINK_FLAGS "${BALL_LINK_FLAGS} -fprofile-arcs")
	ENDIF()
ENDIF()

TARGET_LINK_LIBRARIES(BALL ${BALL_DEP_LIBRARIES})

SET_TARGET_PROPERTIES(BALL PROPERTIES
	COMPILE_FLAGS "${BALL_COMPILE_FLAGS}"
	LINK_FLAGS "${BALL_LINK_FLAGS}"
)

## add BENCHMARK target
ADD_SUBDIRECTORY(source/BENCHMARKS/)


######################################################################
# Installation Routines
######################################################################

IF(NOT APPLE)
	INSTALL(TARGETS BALL
		EXPORT BALLExportGroup
		COMPONENT "${COMPONENT_LIBBALL}"
		RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
		LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
		ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	)

	## If we are not installing into the source directory directly, take
	## care of data files and headers
	IF (NOT "${CMAKE_INSTALL_PREFIX}" STREQUAL ".")
		### Install the data directories
		INSTALL(DIRECTORY data/ DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/BALL COMPONENT ${COMPONENT_LIBBALL})

		### Install the includes
		INSTALL(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT ${COMPONENT_LIBBALL_DEV})
	ENDIF()

	## The configured files have to be copied in any case
	INSTALL(FILES ${PROJECT_BINARY_DIR}/include/BALL/CONFIG/config.h
		DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/BALL/CONFIG"
		COMPONENT ${COMPONENT_LIBBALL_DEV}
	)
ENDIF()



########################################################
# Tests
########################################################

# Tests
ENABLE_TESTING()
ADD_SUBDIRECTORY(test EXCLUDE_FROM_ALL) ## configure the Test project (but do not add it to BALL itself)

# Targets list
INCLUDE(BALLTargets)

# CMake configuration message
MESSAGE(STATUS "")
MESSAGE(STATUS "-----------------------------------------------------------------")
MESSAGE(STATUS "")
MESSAGE(STATUS "You have successfully configured BALL.")
MESSAGE(STATUS "")
MESSAGE(STATUS "For a full list of make targets execute:")
MESSAGE(STATUS "'make targets'")
MESSAGE(STATUS "")
MESSAGE(STATUS "-----------------------------------------------------------------")
MESSAGE(STATUS "")

######################################################
# Create CMake package configurations to be used with
# find_package()for:
#
#   - local build tree BALL
#   - installed BALL
#
######################################################

INCLUDE(CMakePackageConfigHelpers)


# Collect Qt6 dependencies
GET_TARGET_PROPERTY(QT6_LINK_LIBRARIES BALL LINK_LIBRARIES)

SET(QT6_IMPORTED_TARGETS "")
FOREACH(TMP_LINKED_LIB ${QT6_LINK_LIBRARIES})
	IF(${TMP_LINKED_LIB} MATCHES ".*Qt6")
		STRING(REPLACE "::" "" TMP_QT6_LINK_LIBRARY ${TMP_LINKED_LIB})
		SET(QT6_IMPORTED_TARGETS ${QT6_IMPORTED_TARGETS} ${TMP_QT6_LINK_LIBRARY})
	ENDIF()
ENDFOREACH()

# Collect BALL include directories
GET_DIRECTORY_PROPERTY(BALL_INCLUDE_DIRS INCLUDE_DIRECTORIES)


# Generate CMake package configuration for BALL build tree
CONFIGURE_FILE(
	"${PROJECT_SOURCE_DIR}/cmake/BALLConfig.cmake.in"
	"${PROJECT_BINARY_DIR}/cmake/BALLConfig.cmake"
	@ONLY
)

# Generate BALL package config version file
write_basic_package_version_file(
	"${PROJECT_BINARY_DIR}/cmake/BALLConfigVersion.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY AnyNewerVersion
)

# Generate exports
EXPORT(TARGETS BALL
	FILE ${PROJECT_BINARY_DIR}/cmake/BALLExport.cmake)

# Store BALL build directory in the CMake user package registry
EXPORT(PACKAGE BALL)


# Generate CMake package configuration for BALL installation
IF(NOT APPLE)
	# Installation path for BALL CMake package configuration files
	SET(BALL_CMAKE_EXPORT_PATH ${CMAKE_INSTALL_LIBDIR}/cmake/BALL CACHE PATH "Path to the cmake package configuration files")

	LIST(REMOVE_ITEM BALL_INCLUDE_DIRS "${PROJECT_BINARY_DIR}/include")
	LIST(REMOVE_ITEM BALL_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
	SET(BALL_INCLUDE_DIRS ${BALL_PATH}/include/ ${BALL_INCLUDE_DIRS})

	CONFIGURE_FILE(
		"${PROJECT_SOURCE_DIR}/cmake/BALLConfig.cmake.in"
		"${PROJECT_BINARY_DIR}/exports/BALLConfig.cmake"
		@ONLY
	)

	INSTALL(FILES
		"${PROJECT_BINARY_DIR}/exports/BALLConfig.cmake"
		"${PROJECT_BINARY_DIR}/cmake/BALLConfigVersion.cmake"
		DESTINATION "${BALL_CMAKE_EXPORT_PATH}/"
		COMPONENT   ${COMPONENT_LIBBALL_DEV}
	)

	INSTALL(EXPORT BALLExportGroup
		DESTINATION ${BALL_CMAKE_EXPORT_PATH}
		FILE BALLExport.cmake
		COMPONENT "${COMPONENT_LIBBALL_DEV}")
ENDIF()
