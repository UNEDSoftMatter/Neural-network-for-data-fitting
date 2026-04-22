/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <stdio.h>


//--- Function to propagate values through the neural network ---
__global__ void kernel_forward_propagation(int batch,
					   int*   __restrict__ batch_index,
					   real*  __restrict__ data,
					   real** __restrict__ W_hidden,
					   real** __restrict__ b_hidden,
					   real*  __restrict__ W_out,
					   real*  __restrict__ b_out,
					   real** __restrict__ z,
					   real** __restrict__ a,
					   real*  __restrict__ exit_value) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  //-- imax is the maximum number of threads --
  int imax = N_per_batch;
  if (batch == Nbatches - 1)
    imax = Ndata - batch_index[Nbatches - 1];
  if (i >= imax) return;

  int row       = batch_index[batch];
  int start_pos = row*3 + i*3;
  real xdata    = data[start_pos];
  real ydata    = data[start_pos + 1];
  // real zdata    = data[start_pos + 2];

  //--- Propagation through the first hidden layer ---
  // z_j^(k) = sigma_act(sum_i w_ji^(k) * z_i^(k-1) + b_j^(k))
  // indices ranges:
  // w_j0: 0 -> Nneurons - 1
  // w_j1: Nneurons -> Nneurons + (Nneurons - 1)
  // etc
  // In general, w_ji: i*Nneurons -> i*Nneurons + (Nneurons - 1)
  // In general w_ji^k = W_hidden[k][j + i*Nneurons]  
  for (int n = 0; n < Nneurons; n++) {
    a[0][n + i*Nneurons] = W_hidden[0][n           ] * xdata +
                           W_hidden[0][n + Nneurons] * ydata +
                           b_hidden[0][n];

    // In each neuron we store as many values as batches. i is the batch.
    z[0][n + i*Nneurons] = kernel_activation_function(a[0][n + i*Nneurons]);
  }

  //--- Propagation through the rest of hidden layers ---
  for (int k = 1; k < Nhidden; k++)       // Layers
    for (int n = 0; n < Nneurons; n++) {  // Origin neuron

      // The initialization of a is done directly with the bias
      a[k][n + i*Nneurons] = b_hidden[k][n];
      for (int m = 0; m < Nneurons; m++) { //Destiny neuron
	// We add W_nm^(k) * z_m^(k-1)
	a[k][n + i*Nneurons] = a[k][n + i*Nneurons] +
	  W_hidden[k][n + m*Nneurons] * z[k-1][m + i*Nneurons];
      }
      // In each neuron we store as many values as batches. i is the batch.      
      z[k][n + i*Nneurons] = kernel_activation_function(a[k][n + i*Nneurons]);
    }

  //--- Propagation from the last hidden layer to the exit ---
  // exit_val = sum_j Wout_j * z_j + b_out
  exit_value[i] = b_out[0];
  for (int n = 0; n < Nneurons; n++) 
    exit_value[i] = exit_value[i] + W_out[n] * z[Nhidden-1][n + i*Nneurons];
}
