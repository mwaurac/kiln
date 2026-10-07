#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Executor.h>
#include <core/Tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <stdexcept>

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

  m.attr("float32") = kiln::DType::F32;
  m.attr("f16") = kiln::DType::F16;
  m.attr("bf16") = kiln::DType::BF16;
  m.attr("q8_0") = kiln::DType::Q8_0;

  m.def("empty",
      &kiln::Tensor::empty,
      py::arg("shape"),
      py::arg("dtype") = kiln::DType::F32,
      py::arg("device") = kiln::Device::CPU);

  m.def("zeros",
      &kiln::Tensor::zeros,
      py::arg("shape"),
      py::arg("dtype") = kiln::DType::F32,
      py::arg("device") = kiln::Device::CPU);
  m.def("ones",
      &kiln::Tensor::ones,
      py::arg("shape"),
      py::arg("dtype") = kiln::DType::F32,
      py::arg("device") = kiln::Device::CPU);

  m.def("nbytes",
      &kiln::nbytes,
      py::arg("numel"),
      py::arg("dtype"),
      "Block-aware storage size in bytes for `numel` elements of `dtype`.");
  m.def("itemsize",
      &kiln::dtype_itemsize,
      py::arg("dtype"),
      "Scalar element size in bytes. Raises for block-quantized dtypes.");

  py::class_<kiln::Tensor>(m, "Tensor")
      .def(py::init([](std::vector<uint64_t> shape, kiln::DType dtype, kiln::Device device) {
        return kiln::Tensor(shape, dtype, device);
      }),
          py::arg("shape"),
          py::arg("dtype") = kiln::DType::F32,
          py::arg("device") = kiln::Device::CPU)

      .def("__repr__", &kiln::Tensor::print_tensor)

      .def_property_readonly("shape", &kiln::Tensor::shape)
      .def_property_readonly("strides", &kiln::Tensor::strides)
      .def_property_readonly("dtype", &kiln::Tensor::dtype)
      .def_property_readonly("device", &kiln::Tensor::device)
      .def_property_readonly("T", [](const kiln::Tensor &t) { return t.transpose(); })
      .def_property_readonly("mT", [](const kiln::Tensor &t) { return t.transpose(-2, -1); })

      .def_property_readonly("nbytes", &kiln::Tensor::nbytes)
      .def_property_readonly("itemsize", &kiln::Tensor::itemsize)
      .def_property_readonly("has_storage", &kiln::Tensor::has_storage)

      .def("numel", &kiln::Tensor::numel)
      .def("is_contiguous", &kiln::Tensor::is_contiguous)

      .def("data",
          [](const kiln::Tensor &t) {
            if (!t.has_storage()) {
              throw std::runtime_error(
                  "Tensor.data: tensor has no storage (lazy tensor); call execute() first");
            }
            if (!t.is_contiguous()) {
              throw std::runtime_error(
                  "Tensor.data: tensor is non-contiguous; call contiguous() first");
            }
            const std::size_t n = t.nbytes();
            if (n == 0) {
              return py::bytes("", 0);
            }
            return py::bytes(static_cast<const char *>(t.data()), n);
          })

      .def("reshape", &kiln::Tensor::reshape, py::arg("dims"))
      .def("view", &kiln::Tensor::view, py::arg("dims"))
      .def("transpose",
          static_cast<kiln::Tensor (kiln::Tensor::*)(int64_t, int64_t) const>(
              &kiln::Tensor::transpose),
          py::arg("dim0"),
          py::arg("dim1"))
      .def("contiguous", &kiln::Tensor::contiguous);
  m.def(
      "execute",
      [](const kiln::Tensor &output) { kiln::execute(output); },
      py::arg("output"),
      "Execute the graph producing `output`. No-op when `output` has no producer.");
}
