/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <stdio.h>

//--- Function to calculate the average gradients of biases
//    of hidden layers (including the first one)---
__global__ void kernel_average_grad_b_hidden(int  batch,
					     int  k, 
					     int*   __restrict__ batch_index,
					     real** __restrict__ grad_b_hidden,
					     real** __restrict__ delta) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= Nneurons) return;

  //-- We determine the number of data in the batch --
  int Nbatch_data = N_per_batch;
  if (batch == Nbatches - 1)
    Nbatch_data = Ndata - batch_index[Nbatches - 1];

  real sum = 0.0;
  for (int b = 0; b < Nbatch_data; b++){
    // In each neuron we store as many values as batches. b is the batch.
    sum += delta[k][i + b*Nneurons];
  }
  
  grad_b_hidden[k][i] = sum / (real)Nbatch_data;
  

}
