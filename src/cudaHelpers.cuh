#ifndef __CUDAHELPERS_H__
#define __CUDAHELPERS_H__

// Error checking GPU calls
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif /* ifndef __CUDAHELPERS_H__ */
