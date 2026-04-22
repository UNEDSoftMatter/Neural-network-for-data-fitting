/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#ifndef KERNEL_FUNCTIONS_H
#define KERNEL_FUNCTIONS_H

#include "config.h"

// Constants
//-- The kernel constants are declared at kernel_declare_constants.cu --
extern __constant__ int Ndata;
extern __constant__ int Nbatches;
extern __constant__ int N_per_batch;
extern __constant__ int Nneurons;
extern __constant__ int Nhidden;
extern __constant__ real beta1;
extern __constant__ real beta2;
extern __constant__ real eta;
extern __constant__ real epsilon_adam;

// Functions executed by the GPU
__global__ void kernel_print_constants();
__global__ void kernel_forward_propagation(int batch,
					   int*   __restrict__ batch_index,
					   real*  __restrict__ data,
					   real** __restrict__ W_hidden,
					   real** __restrict__ b_hidden,
					   real*  __restrict__ W_out,
					   real*  __restrict__ b_out,
					   real** __restrict__ z,
					   real** __restrict__ a,
					   real*  __restrict__ exit_value);
__global__ void kernel_back_propagation(int batch,
					int*   __restrict__ batch_index,
					real*  __restrict__ data,
					real** __restrict__ W_hidden,
					real*  __restrict__ W_out,
					real** __restrict__ a,
					real** __restrict__ delta,
					real*  __restrict__ delta_out,
					real*  __restrict__ exit_value);
__device__ real kernel_activation_function(real x);
__device__ real kernel_der_activation_function(real x);
__global__ void kernel_average_gradients(int batch,
					 int*   __restrict__ batch_index,
					 real*  __restrict__ data,
					 real** __restrict__ grad_W_hidden,
					 real** __restrict__ grad_b_hidden,
					 real*  __restrict__ grad_W_out,
					 real*  __restrict__ grad_b_out,
					 real** __restrict__ delta,
					 real*  __restrict__ delta_out,
					 real** __restrict__ z);
__global__ void kernel_neural_network_average_gradients(int batch,
							int*   __restrict__ batch_index,
							real*  __restrict__ data,
							real** __restrict__ grad_W_hidden,
							real** __restrict__ grad_b_hidden,
							real*  __restrict__ grad_W_out,
							real*  __restrict__ grad_b_out,
							real** __restrict__ z);
__global__ void kernel_update_ADAM_hidden(int k,
					  int size,
					  int t,
					  real** __restrict__ grad_x_hidden,
					  real** __restrict__ x_hidden,
					  real** __restrict__ m_x_hidden,
					  real** __restrict__ v_x_hidden);
__global__ void kernel_update_ADAM_out(int size,
				       int t,
				       real* __restrict__ grad_x_out,
				       real* __restrict__ x_out,
				       real* __restrict__ m_x_out,
				       real* __restrict__ v_x_out);
__global__ void kernel_loss_function_to_zero(real*  __restrict__ loss_function);
__global__ void kernel_loss_function(int batch,
				     real*  __restrict__ loss_function,
				     real*  __restrict__ exit_value,
				     int*   __restrict__ batch_index,
				     real*  __restrict__ data);
__global__ void kernel_average_grad_W_hidden(int  batch,
					     int  k,
					     int*   __restrict__ batch_index,
					     real** __restrict__ grad_W_hidden,
					     real** __restrict__ delta,
					     real** __restrict__ z);
__global__ void kernel_average_grad_W_hidden_layer0(int  batch,
						    int*   __restrict__ batch_index,
						    real** __restrict__ grad_W_hidden,
						    real** __restrict__ delta,
						    real*  __restrict__ data);
__global__ void kernel_average_grad_W_out(int  batch,
					  int*   __restrict__ batch_index,
					  real*  __restrict__ grad_W_out,
					  real*  __restrict__ delta_out,
					  real** __restrict__ z);
__global__ void kernel_average_grad_b_hidden(int  batch,
					     int  k, 
					     int*   __restrict__ batch_index,
					     real** __restrict__ grad_b_hidden,
					     real** __restrict__ delta);
__global__ void kernel_average_grad_b_out(int  batch,
					  int*  __restrict__ batch_index,
					  real* __restrict__ grad_b_out,
					  real* __restrict__ delta_out);
#endif // KERNEL_FUNCTIONS_H
