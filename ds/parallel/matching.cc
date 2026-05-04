namespace heihgm {
namespace ds {
namespace parallel {
#ifdef CUDA_IS_ENABLED

__global__ void IntToDoubleArray(int* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // The hardware handles the casting here
    output[idx] = (float)input[idx];
  }
}
#endif
}  // namespace parallel
}  // namespace ds
}  // namespace heihgm