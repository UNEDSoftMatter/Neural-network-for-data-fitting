/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include "atomicAdd_double.cuh"

//--- Function to calculate the loss function ---
__global__ void kernel_loss_function(int batch,
				     real*  __restrict__ loss_function,
				     real*  __restrict__ exit_value,
				     int*   __restrict__ batch_index,
				     real*  __restrict__ data) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  //-- imax is the maximum number of threads --
  int imax = N_per_batch;
  if (batch == Nbatches - 1)
    imax = Ndata - batch_index[Nbatches - 1];
  if (i >= imax) return;

  int row       = batch_index[batch];
  int start_pos = row*3 + i*3;  
  real zdata    = data[start_pos + 2];

  real diff = (exit_value[i] - zdata);
  real sum_term = diff * diff/((real)imax);
  
  //-- The sum is saved in loss_function (mean squared error) --  
#if __CUDA_ARCH__ >= 600
  // modern GPUs -> native atomicAdd
  atomicAdd(loss_function, sum_term);
#else
  // old GPUs (< 6.0) -> use CAS emulation
  atomicAdd(loss_function, sum_term);
#endif  


}
