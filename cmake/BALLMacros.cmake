## Add the source files in SOURCES_LIST to the list
## of files compiled into libBALL, and mark them as
## part of source group GROUP
MACRO(ADD_BALL_SOURCES GROUP SOURCES_LIST)
	SET(DIRECTORY source/${GROUP})

	### add full path to the filenames ###
	SET(SOURCES)
	FOREACH(i ${SOURCES_LIST})
		LIST(APPEND SOURCES ${DIRECTORY}/${i})
	ENDFOREACH()

	### pass source file list to the upper instance ###
	SET(BALL_sources ${BALL_sources} ${SOURCES})

	### source group definition ###
	STRING(REGEX REPLACE "/" "\\\\" S_GROUP ${GROUP})
	SOURCE_GROUP("Source Files\\\\${S_GROUP}" FILES ${SOURCES})
ENDMACRO()

## Add the header files in HEADERS_LIST to the list
## of files compiled into libBALL, and mark them as
## part of source group GROUP
MACRO(ADD_BALL_HEADERS GROUP HEADERS_LIST)
	SET(DIRECTORY include/BALL/${GROUP})

	### add full path to the filenames ###
	SET(HEADERS)
	FOREACH(i ${HEADERS_LIST})
		### make sure we do not have absolute paths flying around...
		GET_FILENAME_COMPONENT(i ${i} NAME)
		LIST(APPEND HEADERS ${DIRECTORY}/${i})
	ENDFOREACH()

	### pass source file list to the upper instance ###
	SET(BALL_headers ${BALL_headers} ${HEADERS})

	### source group definition ###
	STRING(REGEX REPLACE "/" "\\\\" S_GROUP ${GROUP})
	SOURCE_GROUP("Header Files\\\\${S_GROUP}" FILES ${HEADERS})
ENDMACRO()

## Add a parser and corresponding lexer to libBALL
MACRO(ADD_BALL_PARSER_LEXER GROUP BASENAME PREFIX)
	SET(DIRECTORY source/${GROUP})

	SET(PARSERINPUT ${DIRECTORY}/${BASENAME}Parser.y)
	SET(LEXERINPUT ${DIRECTORY}/${BASENAME}Lexer.l)

	FILE(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/${DIRECTORY})

	SET(PARSEROUTPUT ${PROJECT_BINARY_DIR}/${DIRECTORY}/${BASENAME}Parser.C)
	SET(PARSERHEADER ${PROJECT_BINARY_DIR}/${DIRECTORY}/${BASENAME}Parser.h)
	SET(LEXEROUTPUT  ${PROJECT_BINARY_DIR}/${DIRECTORY}/${BASENAME}Lexer.C )

	## Oh, what an ugly hack...
	BISON_TARGET(${BASENAME}Parser ${PARSERINPUT} ${PARSEROUTPUT} COMPILE_FLAGS "--defines=${PARSERHEADER} -p${PREFIX}")
	FLEX_TARGET(${BASENAME}Lexer ${LEXERINPUT} ${LEXEROUTPUT} COMPILE_FLAGS "-P${PREFIX}")

	ADD_FLEX_BISON_DEPENDENCY(${BASENAME}Lexer ${BASENAME}Parser)

	SET(BALL_sources ${BALL_sources} ${PARSERINPUT} ${PARSEROUTPUT} ${LEXERINPUT} ${LEXEROUTPUT})

	SOURCE_GROUP("Source Files\\\\${GROUP}" FILES ${PARSEROUTPUT} ${LEXEROUTPUT})
	SOURCE_GROUP("Parser Files\\\\${GROUP}" FILES ${PARSERINPUT})
	SOURCE_GROUP("Lexer Files\\\\${GROUP}" FILES ${LEXERINPUT})
ENDMACRO()
