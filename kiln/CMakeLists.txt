file(GLOB KILN_BINDING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/bindings/*.cpp")

if(KILN_BINDING_SOURCES STREQUAL "")
  message(WARNING "KILN_BUILD_PYTHON=ON but no sources in kiln/bindings/; skipping _kiln extension")
  return()
endif()

pybind11_add_module(_kiln ${KILN_BINDING_SOURCES})
target_link_libraries(_kiln PRIVATE kiln_core)

target_compile_definitions(
  _kiln PRIVATE KILN_VERSION="${PROJECT_VERSION}"
)

# scikit-build-core picks everything installed via install() into the wheel.
install(TARGETS _kiln DESTINATION kiln)
