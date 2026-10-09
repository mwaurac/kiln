add_library(kiln_common INTERFACE)
target_include_directories(kiln_common INTERFACE "${PROJECT_SOURCE_DIR}/csrc")
target_compile_features(kiln_common INTERFACE cxx_std_20)
