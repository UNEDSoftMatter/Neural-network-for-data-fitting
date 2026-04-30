/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <stdio.h>


//--- Function to calculate the average gradients of weights
//    of the output layer ---
__global__ void kernel_average_grad_W_out(int  batch,
					  int*   __restrict__ batch_index,
					  real*  __restrict__ grad_W_out,
					  real*  __restrict__ delta_out,
					  real** __restrict__ z) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= Nneurons) return;

  // indices ranges:
  // w_j0: 0 -> Nneurons - 1
  // w_j1: Nneurons -> Nneurons + (Nneurons - 1)
  // etc
  // In general, w_mn: m*Nneurons -> m*Nneurons + (Nneurons - 1)
  // In general w_mn^k = W_hidden[k][n + m*Nneurons]    

  //-- We determine the number of data in the batch --
  int Nbatch_data = N_per_batch;
  if (batch == Nbatches - 1)
    Nbatch_data = Ndata - batch_index[Nbatches - 1];

  real sum = 0.0;
  for (int b = 0; b < Nbatch_data; b++){
    // In each neuron we store as many values as data in the batch. b is the batch.
    sum += delta_out[b] * z[Nhidden - 1][i + b*Nneurons];
  }
  
  grad_W_out[i] = sum / (real)Nbatch_data;
  
}
