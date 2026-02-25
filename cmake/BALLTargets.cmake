ADD_CUSTOM_TARGET(targets
	COMMAND ${CMAKE_COMMAND} -E echo ""
	COMMAND ${CMAKE_COMMAND} -E echo "=========================================================================="
	COMMAND ${CMAKE_COMMAND} -E echo ""
	COMMAND ${CMAKE_COMMAND} -E echo "The following make targets are available:"
	COMMAND ${CMAKE_COMMAND} -E echo ""
	COMMAND ${CMAKE_COMMAND} -E echo "    BALL            builds the BALL library"
	COMMAND ${CMAKE_COMMAND} -E echo "    build_tests     builds the unit tests"
	COMMAND ${CMAKE_COMMAND} -E echo ""
	COMMAND ${CMAKE_COMMAND} -E echo "=========================================================================="
	COMMAND ${CMAKE_COMMAND} -E echo ""
	COMMENT "The most important targets for BALL"
	VERBATIM
)
