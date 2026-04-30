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
//    of the first hidden layer ---
__global__ void kernel_average_grad_W_hidden_layer0(int  batch,
						    int*   __restrict__ batch_index,
						    real** __restrict__ grad_W_hidden,
						    real** __restrict__ delta,
						    real*  __restrict__ data) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= 2 * Nneurons) return; // 2 inputs

  // indices ranges:
  // w_j0: 0 -> Nneurons - 1
  // w_j1: Nneurons -> Nneurons + (Nneurons - 1)
  // etc
  // In general, w_mn: m*Nneurons -> m*Nneurons + (Nneurons - 1)
  // In general w_mn^k = W_hidden[k][n + m*Nneurons]    
  int n = i % Nneurons;      // destiny neuron
  int m = i / Nneurons;      // origin neuron

  //-- We determine the number of data in the batch --
  int Nbatch_data = N_per_batch;
  if (batch == Nbatches - 1)
    Nbatch_data = Ndata - batch_index[Nbatches - 1];
  int row = batch_index[batch];

  real sum = 0.0;
  for (int b = 0; b < Nbatch_data; b++){
    // The input data es determined
    int start_pos = 3*(row + b);
    real input_data;
    if (m == 0)
      input_data = data[start_pos]; //xdata
    else // m == 1
      input_data = data[start_pos + 1];	  //ydata
    
    // In each neuron we store as many values as data in the batch. b is the batch.    
    sum += delta[0][n + b*Nneurons] * input_data;
  }

    
  grad_W_hidden[0][i] = sum / (real)Nbatch_data;

}
