##
#  CMake script copied from the step-1 tutorial program:
##

SET(TARGET "simulation")
SET(TARGET_SRC
  ${TARGET}.cpp
  )


CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS")
FIND_PACKAGE(deal.II 8.4 QUIET
  HINTS ${deal.II_DIR} $
  SET(TARGET "simulation")
  SET(TARGET_SRC
    ${TARGET}.cpp
    )
  
  CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
  
  FIND_PACKAGE(deal.II 8.4 QUIET
    HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
    )
  IF(NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
      "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
      "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
      "or set an environment variable \"DEAL_II_DIR\" that contains this path."
      )
  ENDIF()
  
  DEAL_II_INITIALIZE_CACHED_VARIABLES()
  PROJECT(${TARGET})
  DEAL_II_INVOKE_AUTOPILOT(){DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()