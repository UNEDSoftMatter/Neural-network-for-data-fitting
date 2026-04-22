/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <stdio.h>


//--- Function to calculate the average gradients of weights and biases of the neural
//    network
__global__ void kernel_average_gradients(int batch,
					 int*   __restrict__ batch_index,
					 real*  __restrict__ data,
					 real** __restrict__ grad_W_hidden,
					 real** __restrict__ grad_b_hidden,
					 real*  __restrict__ grad_W_out,
					 real*  __restrict__ grad_b_out,
					 real** __restrict__ delta,
					 real*  __restrict__ delta_out,
					 real** __restrict__ z) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i == 0) { // Done only by a thread   
    //-- N_batch_data is the number of data in the batch --
    int N_batch_data = N_per_batch;
    if (batch == Nbatches - 1)
      N_batch_data = Ndata - batch_index[Nbatches - 1];
    real N_batch_data_inv = 1.0 / (real)N_batch_data;
    
    //--- Gradient averages of w in the first hidden layer ---
    // z_j^(k) = sigma_act(sum_i w_ji^(k) * z_i^(k-1) + b_j^(k))
    // indices ranges:
    // w_j0: 0 -> Nneurons - 1
    // w_j1: Nneurons -> Nneurons + (Nneurons - 1)
    // etc
    // In general, w_ji: i*Nneurons -> i*Nneurons + (Nneurons - 1)
    // In general w_ji^k = W_hidden[k][j + i*Nneurons]
    // In the first layer w_ji -> j: Nneurons, i: N_inputs
    int row = batch_index[batch];
    for (int n = 0; n < Nneurons; n++) {    // Neurons in first layer
      //w_{nm} -> w_{n0}  y  w_{n1}
      grad_W_hidden[0][n]            = 0;
      grad_W_hidden[0][n + Nneurons] = 0;
      for (int b = 0; b < N_batch_data; b++) {   // N_batch_data
	int start_pos = 3*(row + b);
	real xdata = data[start_pos];
	real ydata = data[start_pos + 1];	
	grad_W_hidden[0][n] = grad_W_hidden[0][n] +
	  delta[0][n + b*Nneurons] *  xdata;
	grad_W_hidden[0][n + Nneurons] = grad_W_hidden[0][n + Nneurons] +
	  delta[0][n + b*Nneurons] *  ydata;
      }
      grad_W_hidden[0][n] = grad_W_hidden[0][n] * N_batch_data_inv;
      grad_W_hidden[0][n + Nneurons] =
	grad_W_hidden[0][n + Nneurons] * N_batch_data_inv;
    }

    //--- Gradient averages of w in the other hidden layers ---
    for (int k = 1; k < Nhidden; k++)         // Hidden layers
      for (int n = 0; n < Nneurons; n++) {    // Neurons in layer k
	for (int m = 0; m < Nneurons; m++) {  // Neurons in layer k-1
	  //w_{nm}
	  grad_W_hidden[k][n + m*Nneurons] = 0;

	  for (int b = 0; b < N_batch_data; b++) {   // N_batch_data
	    grad_W_hidden[k][n + m*Nneurons] =
	      grad_W_hidden[k][n + m*Nneurons] +
	      delta[k][n + b*Nneurons] * z[k-1][m + b*Nneurons];
	  }
	    grad_W_hidden[k][n + m*Nneurons] =
	      grad_W_hidden[k][n + m*Nneurons] * N_batch_data_inv;
	}
      }

    //--- Gradient averages of w in the output layer ---
    for (int n = 0; n < Nneurons; n++){  // Neurons
      grad_W_out[n] = 0;
      for (int b = 0; b < N_batch_data; b++)    // N_batch_data	
	grad_W_out[n] = grad_W_out[n] +
	  delta_out[b] * z[Nhidden - 1][n + b*Nneurons];
      grad_W_out[n] = grad_W_out[n] * N_batch_data_inv;
    }

    //-- Gradient averages of b in the hidden layers (including input layer)
    for (int k = 0; k < Nhidden; k++)         // Hidden layers
      for (int n = 0; n < Nneurons; n++) {    // Neurons in layer k
	grad_b_hidden[k][n] = 0;
	for (int b = 0; b < N_batch_data; b++) {   // N_batch_data
	  grad_b_hidden[k][n] = grad_b_hidden[k][n] +
	    delta[k][n + b*Nneurons];
	}
	grad_b_hidden[k][n] = grad_b_hidden[k][n] * N_batch_data_inv;
      }

    //-- Gradient averages of b in the output layer (One only exit)
    grad_b_out[0] = 0;
    for (int b = 0; b < N_batch_data; b++)   // N_batch_data
      grad_b_out[0] = grad_b_out[0] + delta_out[b];
    grad_b_out[0] = grad_b_out[0] * N_batch_data_inv;
    
  } // i = 0
	

}
