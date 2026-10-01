cmake_minimum_required(VERSION 3.20)
project(kiln VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(KILN_SOURCES
csrc/core/Dtype.cpp
)

add_library(kiln_core STATIC
${KILN_SOURCES}
)

target_include_directories(kiln_core PUBLIC "${PROJECT_SOURCE_DIR}/csrc")
set_target_properties(kiln_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

include (cmake/Dependencies.cmake)
kiln_resolve_dependencies()

if (KILN_BUILD_PYTHON)
  add_subdirectory(kiln)
endif()

if(KILN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

include(cmake/Summary.cmake)
kiln_print_summary()

