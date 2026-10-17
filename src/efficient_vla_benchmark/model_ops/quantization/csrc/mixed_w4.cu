// First-party C ABI wrapper around pinned CUTLASS 3.5.1 (BSD-3-Clause).
// No Torch ABI, persistent workspace, weight reconstruction or implicit stream.
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>

using Int4Gemm = cutlass::gemm::device::Gemm<
    cutlass::int4b_t, cutlass::layout::RowMajor,
    cutlass::int4b_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor, int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 128>,
    cutlass::gemm::GemmShape<32, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    cutlass::epilogue::thread::LinearCombinationClamp<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

extern "C" int mixed_w4_abi() { return 1; }

extern "C" int int4_gemm(const void *a, const void *b, void *c,
                         int m, int n, int k, void *stream) {
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0 || n % 8 || k % 32) return -1;
    Int4Gemm::Arguments args(
        {m, n, k},
        {static_cast<const cutlass::int4b_t*>(a), k},
        {static_cast<const cutlass::int4b_t*>(b), k},
        {static_cast<const int32_t*>(c), n},
        {static_cast<int32_t*>(c), n}, {1, 0});
    auto status = Int4Gemm::can_implement(args);
    if (status != cutlass::Status::kSuccess) return static_cast<int>(status);
    if (Int4Gemm::get_workspace_size(args) != 0) return -2;
    Int4Gemm operation;
    return static_cast<int>(operation(args, nullptr, static_cast<cudaStream_t>(stream)));
}
