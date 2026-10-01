#include <core/Dtype.h>
#include <pybind11/pybind11.h>

#include "core/Device.h"

namespace py = pybind11;

PYBIND11_MODULE(_kiln, m) {
  m.doc() = "Kiln doc";
#ifdef KILN_VERSION
  m.attr("__version__") = KILN_VERSION;
#else
  m.attr("__version__") = "0.1.0";
#endif

  py::enum_<kiln::DType>(m, "DType")
      .value("F32", kiln::DType::F32)
      .value("F16", kiln::DType::F16)
      .value("BF16", kiln::DType::BF16)
      .value("Q8_0", kiln::DType::Q8_0)
      .export_values();

  py::enum_<kiln::Device>(m, "Device")
      .value("CPU", kiln::Device::CPU)
      .value("CUDA", kiln::Device::CUDA)
      .export_values();
}
