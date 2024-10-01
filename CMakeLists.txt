cmake_minimum_required( VERSION 2.8.3 )
if( POLICY CMP0048 )
  # Silence CMP0048 warning about missing project VERSION.
  cmake_policy(SET CMP0048 NEW)
endif()
if( POLICY CMP0042 )
  cmake_policy(SET CMP0042 NEW)
endif()
project( t22_flow )
set(version_major 0)
set(version_minor 1)
set(version_patch 0)
set(version "${version_major}.${version_minor}.${version_patch}")
set(package-contact "skaiua75@gmail.com")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")

set(${PROJECT_NAME}_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/include
    CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

add_subdirectory( external/PX4-OpticalFlow OpticalFlow )
set( OpticalFlow_LIBS "OpticalFlow" )


set(t22_flow_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpticalFlow_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

include_directories(t22_flow_INCLUDE_DIRS)

add_executable(t22_flow main.cpp)

target_link_libraries( t22_flow
  ${OpenCV_LIBS}
  ${OpticalFlow_LIBS}
)


target_include_directories(t22_flow PUBLIC include)

# Generate the Config file from the .in file, doing @VAR@ substitution only
configure_file(t22_flowConfig.cmake.in t22_flowConfig.cmake @ONLY)

# INSTALL STEPS
INSTALL(
  TARGETS t22_flow
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib)

INSTALL(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/PX4-OpticalFlow/include/
  DESTINATION include)

INSTALL(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION include)

INSTALL(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/t22_flowConfig.cmake
  DESTINATION ${CMAKE_INSTALL_PREFIX})
