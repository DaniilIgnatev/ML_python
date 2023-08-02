import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# Define CUDA function (Kernel)
mod = SourceModule("""
    __global__ void multiply_them(float *dest, float *a, float *b)
    {
      const int i = threadIdx.x;
      dest[i] = a[i] * b[i];
    }
""")

# Create the function
multiply_them = mod.get_function("multiply_them")

# Initialize array size
N = 400

# Create arrays
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)

# Allocate memory for output
dest = np.zeros_like(a)


# Call the function
multiply_them(
    drv.Out(dest), drv.In(a), drv.In(b),
    block=(N, 1, 1), grid=(1, 1)
)

# Print the output
print(dest)