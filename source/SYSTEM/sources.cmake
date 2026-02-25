### list all filenames of the directory here ###
SET(SOURCES_LIST
	binaryFileAdaptor.C
	directory.C
	file.C
	fileSystem.C
	mutex.C
	path.C
	simpleDownloader.C
	sysinfo.C
	timer.C
)

ADD_BALL_SOURCES("SYSTEM" "${SOURCES_LIST}")
