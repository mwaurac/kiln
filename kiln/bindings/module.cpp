#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Executor.h>
#include <core/Layout.h>
#include <core/Tensor.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <bit>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {
bool is_native_byteorder(char order) {
  if (order == '=' || order == '|') {
    return true;
  }
  if constexpr (std::endian::native == std::endian::little) {
    return order == '<';
  } else {
    return order == '>';
  }
}

// Wrap `obj` in a type-erased shared owner so C++ storage can keep a Python
// object (ndarray, buffer, mmap) alive. The GIL is re-acquired to drop the
// last reference, since storage may die on a non-Python thread.
std::shared_ptr<void> keep_alive(py::object obj) {
  struct Holder {
    py::object obj;
  };
  auto holder = std::shared_ptr<Holder>(new Holder{std::move(obj)}, [](Holder *p) {
    const py::gil_scoped_acquire gil;
    delete p;
  });
  return std::shared_ptr<void>(holder, holder.get());
}
}  // namespace

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

  m.def(
      "from_numpy",
      [](py::array arr) {
        const py::dtype dt = arr.dtype();
        if (dt.kind() != 'f' || dt.itemsize() != 4 || !is_native_byteorder(dt.byteorder())) {
          throw std::runtime_error(
              "from_numpy: only native float32 arrays are supported, got dtype " +
              py::str(dt).cast<std::string>());
        }
        if (!arr.writeable()) {
          throw std::runtime_error(
              "from_numpy: array shares memory with the tensor, so it must be writeable; pass a "
              "writeable copy (arr.copy())");
        }
        const py::ssize_t ndim = arr.ndim();
        kiln::Shape shape;
        shape.reserve(static_cast<std::size_t>(ndim));
        for (py::ssize_t d = 0; d < ndim; ++d) {
          const py::ssize_t dim = arr.shape(d);
          if (dim < 0) {
            throw std::runtime_error("from_numpy: negative dimension");
          }
          shape.push_back(static_cast<std::uint64_t>(dim));
        }
        kiln::Strides strides;
        strides.reserve(static_cast<std::size_t>(ndim));
        for (py::ssize_t d = 0; d < ndim; ++d) {
          const py::ssize_t byte_stride = arr.strides(d);
          if (byte_stride < 0) {
            throw std::runtime_error(
                "from_numpy: negative strides cannot be shared; pass a contiguous copy "
                "(np.ascontiguousarray(arr))");
          }
          if (byte_stride % 4 != 0) {
            throw std::runtime_error("from_numpy: byte stride is not a multiple of 4");
          }
          strides.push_back(static_cast<std::uint64_t>(byte_stride / 4));
        }
        return kiln::Tensor::from_blob(arr.mutable_data(),
            shape,
            strides,
            kiln::DType::F32,
            kiln::Device::CPU,
            keep_alive(py::object(arr)));
      },
      py::arg("array"),
      "Wrap a NumPy array as a Kiln tensor. Zero copy");

  m.def(
      "from_buffer",
      [](py::buffer buf,
          std::vector<std::uint64_t> shape,
          kiln::DType dtype,
          std::optional<std::vector<std::uint64_t>> strides) {
        const py::buffer_info info = buf.request();
        if (info.readonly) {
          throw std::runtime_error(
              "from_buffer: buffer is read-only and cannot be shared mutably; pass a writeable "
              "buffer (bytearray, writeable ndarray/memoryview)");
        }
        kiln::Shape kshape(shape.begin(), shape.end());
        kiln::Strides kstrides;
        if (strides) {
          if (strides->size() != kshape.size()) {
            throw std::runtime_error("from_buffer: strides rank does not match shape rank");
          }
          kstrides.assign(strides->begin(), strides->end());
        } else {
          kstrides = kiln::Layout::contiguous(kshape).strides_vec();
        }
        std::size_t numel = kiln::Layout(kshape, kstrides).numel();
        const std::size_t need = kiln::nbytes(numel, dtype);
        const std::size_t have =
            static_cast<std::size_t>(info.size) * static_cast<std::size_t>(info.itemsize);
        if (have < need) {
          throw std::runtime_error("from_buffer: buffer holds " + std::to_string(have) +
                                   " bytes but shape/dtype need " + std::to_string(need));
        }
        return kiln::Tensor::from_blob(info.ptr,
            kshape,
            kstrides,
            dtype,
            kiln::Device::CPU,
            keep_alive(py::object(buf)));
      },
      py::arg("buffer"),
      py::arg("shape"),
      py::arg("dtype") = kiln::DType::F32,
      py::arg("strides") = std::optional<std::vector<std::uint64_t>>{},
      "Wrap a writeable buffer-protocol object as a Kiln tensor without copying.");

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

      .def(
          "numpy",
          [](const kiln::Tensor &t) {
            if (!t.has_storage()) {
              throw std::runtime_error(
                  "Tensor.numpy: tensor has no storage (lazy output); call execute() first");
            }
            if (t.dtype() != kiln::DType::F32) {
              throw std::runtime_error(
                  "Tensor.numpy: only float32 tensors are supported, got "
                  "dtype " +
                  kiln::dtype_name(t.dtype()));
            }
            const kiln::Tensor c = t.contiguous();
            const kiln::Shape kshape = c.shape();
            std::vector<py::ssize_t> shape(kshape.begin(), kshape.end());
            py::array_t<float> out(shape);
            const std::size_t n = c.nbytes();
            if (n > 0) {
              std::memcpy(out.mutable_data(), c.data(), n);
            }
            return out;
          },
          "Copy this tensor to a NumPy array (float32 only).\n")

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
