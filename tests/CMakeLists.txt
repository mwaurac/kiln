include(GoogleTest)

set(KILN_TEST_SOURCES
  cpp/test_storage.cpp
  cpp/test_tensor.cpp
  cpp/test_graph.cpp
  cpp/test_ops.cpp
)

add_executable(kiln_tests ${KILN_TEST_SOURCES})
target_include_directories(kiln_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(kiln_tests PRIVATE kiln_core GTest::gtest_main)
gtest_discover_tests(kiln_tests)
