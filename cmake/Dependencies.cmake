
include(FetchContent)

# PYBIND11
function(kiln_setup_pybind11)
  set(_pybind_dir "${PROJECT_SOURCE_DIR}/third_party/pybind11")

  if(EXISTS "${_pybind_dir}/CMakeLists.txt")
    message(STATUS "pybind11: using submodule at third_party/pybind11")
    add_subdirectory("${_pybind_dir}" pybind11 EXCLUDE_FROM_ALL)
  else()
    message(STATUS "pybind11: submodule not found, fetching via FetchContent")
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
  endif()
endfunction()

# googletest
function(kiln_setup_gtest)
  # For Windows: prevent overriding the parent project's compiler/linker settings.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  set(_gtest_dir "${PROJECT_SOURCE_DIR}/third_party/googletest")

  if(EXISTS "${_gtest_dir}/CMakeLists.txt")
    message(STATUS "googletest: using submodule at third_party/googletest")
    add_subdirectory("${_gtest_dir}" googletest EXCLUDE_FROM_ALL)
  else()
    message(STATUS "googletest: submodule not found, fetching via FetchContent")
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/v1.14.0.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)
  endif()
endfunction()

macro(kiln_resolve_dependencies)
  if(KILN_BUILD_TESTS)
    kiln_setup_gtest()
  endif()
  if(KILN_BUILD_PYTHON)
    kiln_setup_pybind11()
  endif()
endmacro()
