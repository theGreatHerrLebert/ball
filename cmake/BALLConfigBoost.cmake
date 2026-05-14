# Find and configure Boost library
# Use Boost's own BoostConfig.cmake (config mode) for modern Boost (>= 1.70)
IF(POLICY CMP0167)
	CMAKE_POLICY(SET CMP0167 NEW)
ENDIF()

# Mandatory boost components
# Note: 'system' is header-only since Boost 1.69 and no longer a findable
# compiled component in recent Boost CMake configs.
SET(BALL_BOOST_COMPONENTS
	chrono
	date_time
	iostreams
	regex
	serialization
	thread
)

# Detailed messaging in case of failures
SET(Boost_DETAILED_FAILURE_MSG ON)

# Invoke Boost's config-mode package
FIND_PACKAGE(Boost 1.70 REQUIRED COMPONENTS ${BALL_BOOST_COMPONENTS})

# Provide the variables the rest of the BALL build expects
IF(NOT Boost_LIBRARIES)
	SET(Boost_LIBRARIES
		Boost::chrono
		Boost::date_time
		Boost::iostreams
		Boost::regex
		Boost::serialization
		Boost::thread
		Boost::system
	)
ENDIF()
