# Find and configure Boost library
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

# Invoke CMake FindBoost module
FIND_PACKAGE(Boost 1.55 REQUIRED COMPONENTS ${BALL_BOOST_COMPONENTS})
