/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <stdio.h>


//--- Function to calculate delta through back propagation 
__global__ void kernel_back_propagation(int batch,
					int*   __restrict__ batch_index,
					real*  __restrict__ data,
					real** __restrict__ W_hidden,
					real*  __restrict__ W_out,
					real** __restrict__ a,
					real** __restrict__ delta,
					real*  __restrict__ delta_out,
					real*  __restrict__ exit_value) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  //-- imax is the maximum number of threads --
  int imax = N_per_batch;
  if (batch == Nbatches - 1)
    imax = Ndata - batch_index[Nbatches - 1];
  if (i >= imax) return;

  int row = batch_index[batch];
  int start_pos = row*3 + i*3;
  real zdata = data[start_pos + 2];

  // z_j^(k) = sigma_act(sum_i w_ji^(k) * z_i^(k-1) + b_j^(k))
  // indices ranges:  
  // w_j0: 0 -> Nneurons - 1
  // w_j1: Nneurons -> Nneurons + (Nneurons - 1)
  // etc
  // In general, w_ji: i*Nneurons -> i*Nneurons + (Nneurons - 1)
  // In general w_ji^k = W_hidden[k][j + i*Nneurons]
  //-- Back propagation through the exit layer --
  // One output
  delta_out[i] = (exit_value[i] - zdata);

  //-- Back propagation through the last hidden layer --
  for (int n = 0; n < Nneurons; n++)  // Neurons
    delta[Nhidden - 1][n + i*Nneurons] = delta_out[i] * W_out[n] *
      kernel_der_activation_function(a[Nhidden - 1][n + i*Nneurons]);

  //-- Back propagation through the rest of hidden layers --
  // Note that to calculate deltas of the 0 layer we do not need the weights of that layer.
  for (int k = Nhidden-2; k >= 0; k--) {      // Layers
    // We are calculating delta_m^k
    for (int m = 0; m < Nneurons; m++) {  // Neurons 
      real sum_aux = 0;
      for (int n = 0; n < Nneurons; n++)  // Neurons
	sum_aux = sum_aux + delta[k+1][n + i*Nneurons] * W_hidden[k+1][n + m*Nneurons];
      delta[k][m + i*Nneurons] =
	kernel_der_activation_function(a[k][m + i*Nneurons]) * sum_aux;
    }
  }

}
