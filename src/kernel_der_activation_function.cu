/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"

// Derivative of the activation function of the neural network
__device__ real kernel_der_activation_function(real x) {
  
  real der_f;

  //------ ReLU function -----
  // if (x > 0)
  //   der_f = 1.0;
  // else
  //   der_f = 0;

  // //--- Sigmoid function ---
  real sigmoid =  1.0/(1.0 + exp(-x));
  der_f = sigmoid * (1.0 - sigmoid);
  
  return der_f;
}
