#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace kiln {

class Error : public std::runtime_error {
 public:
  Error(const std::string &msg, const char *file, int line, const char *func)
      : std::runtime_error(msg + " (at " + file + ":" + std::to_string(line) + " in " + func + ")"),
        user_msg_(msg),
        file_(file),
        line_(line),
        func_(func) {}

  const std::string &user_message() const noexcept {
    return user_msg_;
  }
  const char *file() const noexcept {
    return file_;
  }
  int line() const noexcept {
    return line_;
  }
  const char *func() const noexcept {
    return func_;
  }

 private:
  std::string user_msg_;
  const char *file_;
  int line_;
  const char *func_;
};

namespace detail {
template <typename... Args>
std::string str_cat(Args &&...args) {
  ::std::ostringstream oss;
  (oss << ... << args);
  return oss.str();
}

template <typename... Args>
[[noreturn]] void fail_at(const char *file, int line, const char *func, Args &&...args) {
  throw ::kiln::Error(str_cat(std::forward<Args>(args)...), file, line, func);
}
}  // namespace detail

}  // namespace kiln

#define KILN_ERROR(...) ::kiln::detail::fail_at(__FILE__, __LINE__, __func__, __VA_ARGS__)

#define KILN_CHECK(cond, ...)           \
  do {                                  \
    if (!(cond)) {                      \
      ::kiln::detail::fail_at(__FILE__, \
          __LINE__,                     \
          __func__,                     \
          "check failed: " #cond ": ",  \
          __VA_ARGS__);                 \
    }                                   \
  } while (0)
