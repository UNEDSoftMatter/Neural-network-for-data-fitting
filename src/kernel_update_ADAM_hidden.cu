/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <stdio.h>

//--- Parameters of the ADAM method are updated, and gradient descent is done.
// All the weights and biases of the corresponding hidden layers are updated.
// size depends on the layer and on whether they are weights or biases:
//-- Weights --
// k = 0 -> size = Nneurons * 2
// k in [1, Nlayers - 1] -> size = Nneurons * Nneurons
//-- biases --
// all layers -> size = Nneurons
__global__ void kernel_update_ADAM_hidden(int k, //In the arguments, x = W or x = m
					  int size,
					  int t,
					  real** __restrict__ grad_x_hidden,
					  real** __restrict__ x_hidden,
					  real** __restrict__ m_x_hidden,
					  real** __restrict__ v_x_hidden) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= size) return;

  //--------- Updating m_x, v_x and x_hidden ---------------
  //-- Update from previous values --
  real g   = grad_x_hidden[k][i];
  real m_x = m_x_hidden[k][i];
  real v_x = v_x_hidden[k][i];
  m_x      = beta1 * m_x + (1.0 - beta1) * g;
  v_x      = beta2 * v_x + (1.0 - beta2) * g*g;

  //-- Bias correction ---
  real m_x_hat = m_x / (1 - pow(beta1, t));
  real v_x_hat = v_x / (1 - pow(beta2, t));

  //-- m_x and v_x are updated:
  m_x_hidden[k][i] = m_x;
  v_x_hidden[k][i] = v_x;

  //-- Weights and biases are corrected --
  x_hidden[k][i] = x_hidden[k][i] - eta * m_x_hat / (sqrt(v_x_hat) + epsilon_adam);  

}
