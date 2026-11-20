#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <executorch/extension/tensor/tensor_ptr.h>

#include "dtype.hpp"
#include "status.hpp"

namespace etrb {

struct ShapeInfo {
  std::vector<executorch::aten::SizesType> sizes;
  size_t numel = 0;
  size_t nbytes = 0;
};

// Overflow-safe validation of shape dims (0 <= d <= INT32_MAX), numel, and nbytes.
StatusOr<ShapeInfo> validate_shape(
    const std::vector<int64_t>& raw_dims,
    executorch::aten::ScalarType dtype) noexcept;

// Allocate raw uninitialized byte storage (no redundant zero-fill before memcpy)
// aligned to at least 64 bytes for SIMD/XNNPACK friendliness.
struct AlignedBuffer {
  void* ptr = nullptr;
  size_t nbytes = 0;

  AlignedBuffer() = default;
  ~AlignedBuffer() noexcept;

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  AlignedBuffer(AlignedBuffer&& other) noexcept;
  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

  static StatusOr<AlignedBuffer> allocate(size_t nbytes) noexcept;
  static StatusOr<AlignedBuffer> allocate_zeroed(size_t nbytes) noexcept;
};

// Check whether an ExecuTorch tensor is laid out in standard C-contiguous
// row-major order (matching logical shape without transposition/gaps).
bool is_standard_contiguous(const executorch::aten::Tensor& tensor) noexcept;

// Copy any ExecuTorch tensor (including non-contiguous / NHWC dim_order tensors)
// into a flat C-contiguous byte buffer of size `tensor.numel() * element_size`.
Status copy_to_contiguous_bytes(
    const executorch::aten::Tensor& tensor,
    void* dst,
    size_t dst_nbytes) noexcept;

// Clone an ExecuTorch tensor into an owned, C-contiguous TensorPtr.
StatusOr<executorch::extension::TensorPtr> clone_contiguous_tensor_ptr(
    const executorch::aten::Tensor& tensor) noexcept;

// Build an owned C-contiguous TensorPtr taking ownership of `buf`.
StatusOr<executorch::extension::TensorPtr> make_owned_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType dtype,
    AlignedBuffer buf) noexcept;

// Build an unowned view TensorPtr pointing into external memory (e.g. a pinned
// Ruby IO::Buffer). Caller guarantees `data` remains valid while the TensorPtr
// is used.
StatusOr<executorch::extension::TensorPtr> make_borrowed_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType dtype,
    void* data) noexcept;

}  // namespace etrb
