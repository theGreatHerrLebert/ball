# Find and configure Boost library
SET(BALL_BOOST_COMPONENTS
	chrono
	date_time
	iostreams
	regex
	serialization
	thread
)
SET(BALL_BOOST_MIN_VERSION 1.70)

# Detailed messaging in case of failures
SET(Boost_DETAILED_FAILURE_MSG ON)

# Invoke CMake FindBoost module
FIND_PACKAGE(Boost ${BALL_BOOST_MIN_VERSION} REQUIRED COMPONENTS ${BALL_BOOST_COMPONENTS})
