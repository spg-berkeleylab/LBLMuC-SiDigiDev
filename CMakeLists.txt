CMAKE_MINIMUM_REQUIRED(VERSION 3.15 FATAL_ERROR)

PROJECT(LBLMuC-SiDigiDev)

INCLUDE("/opt/ilcsoft/muonc/ILCSoft.cmake")

#
# External packages
FILE(GLOB exts RELATIVE ${CMAKE_CURRENT_LIST_DIR} exts/*)
FOREACH(ext ${exts})
  ADD_SUBDIRECTORY(${ext})
ENDFOREACH()

#
# Find and add all packages
FILE(GLOB packages RELATIVE ${CMAKE_CURRENT_LIST_DIR} packages/*)
FOREACH(package ${packages})
  IF(IS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${package})
    MESSAGE("Add package ${package}")
    ADD_SUBDIRECTORY(${package})
  ENDIF()
ENDFOREACH()
