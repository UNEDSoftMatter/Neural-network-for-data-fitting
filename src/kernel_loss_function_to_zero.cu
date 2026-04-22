/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"


//--- Function to initialize loss_function --
__global__ void kernel_loss_function_to_zero(real*  __restrict__ loss_function) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i == 0) loss_function[0] = 0.0;

}
