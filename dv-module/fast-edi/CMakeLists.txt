# Init CMake (require at least version 3.10.0)
CMAKE_MINIMUM_REQUIRED(VERSION 3.10.0)


# Project name and version
PROJECT(fast-edi
	VERSION 1.0.0
	LANGUAGES C CXX)

# Define installation paths
INCLUDE(GNUInstallDirs)
# find_package(Boost REQUIRED COMPONENTS filesystem)
FIND_PACKAGE(dv-processing 1.7.2 REQUIRED)
FIND_PACKAGE(dv 1.6.2 REQUIRED)


# Compile this module (standard C++ file, no additional dependencies)
ADD_LIBRARY(FEDI SHARED FEDI.cpp)

SET_TARGET_PROPERTIES(FEDI
	PROPERTIES
	PREFIX "fedi_"
)

TARGET_LINK_LIBRARIES(FEDI PRIVATE dv::sdk)

INSTALL(TARGETS FEDI DESTINATION ${DV_MODULES_DIR})
